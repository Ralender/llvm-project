//===- GVN.cpp - Global Value Numbering -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an optimistic hash based GVN similar to llvm's
// NewGVN. But it uses different data structures:
//  - Expressions keep pointers on there original value, current representation
//    and current class so there is no need for hash map to keep track of it
//  - Expressions are kept in a intrusive list with other expressions in the
//    same congruence class this way moving expressions between classes (which
//    is very frequent) is cheap.
//  - Expression and there operands are allocated in one consecutive block
//  - The leader of a congruence class is the first expressions in the intrusive
//    list.
// It has a few limitation:
//  - /!\\ It is greatly under tested for now /!\\.
//  - It does not yet handle associative operations like it should.
//  - The leader selection is cheap but often not optimal.
//  - IR editing is still basic. no sinking, hoisting or localized replacement.
//  - Self validation should be pushed further
//  - It deal with memory pessimistically. assuming every operation that
//    accesses memory cannot match with anything other then it self.
//  - for now GVN only deduces global informations. not informations that are
//    true for only part of the cfg
//  - Lacking phi folding like: phi(op, op) = op(phi, phi)
//  - A new expression is created every time an Value/Operation is process.
//    And it is very hard to know that an expression is not used
//    anymore, because expression use other expression without any use tracking
//    or ref-counting. so expressions are not deleted until the GVN is done.
//    this means that memory usage might be high when GVN needs many iterations
//    until it reaches the fixed-point.
//  - GVN currently analyses regions one by one and not all together.
//    This maybe suboptimal. This is because GVN requires a full CFG to operate.
//    And it feels weird build a virtual CFG over structured control-flow when
//    this is literal a lowering step that will be performed.
//
// It could be a good idea to split GVN into the Analysis part that find the
// congruence classes and the IR editing part based of the found classes
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/GvnOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ArrayRecycler.h"

namespace mlir {
#define GEN_PASS_DEF_GVN
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mlir-gvn"

using namespace mlir;

namespace {

thread_local bool isUsingProperHash;

struct BlockEdge {
  Block *from;
  Block *to;
};

using ValOrOp = PointerUnion<Value, Operation *>;

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, ValOrOp voo) {
  if (auto val = voo.dyn_cast<Value>())
    os << val.getImpl() << ":" << val;
  else
    os << voo.dyn_cast<Operation *>() << ":" << *voo.dyn_cast<Operation *>();
  return os;
}

Block *getBlock(ValOrOp voo) {
  if (auto val = voo.dyn_cast<Value>())
    return val.getParentBlock();
  return voo.dyn_cast<Operation *>()->getBlock();
}

struct CongruenceClass;
class Expr;
class ExternalExpr;
class PHIExpr;
class GenericOpExpr;
class CostumeExpr;
class ConstExpr;
class DeadExpr;

/// This is only used for debug. its stores IDs that can be printed when
/// referring to expressions and congruence classes
class WithID {
#ifndef NDEBUG
  unsigned id = 0;
#endif

public:
  unsigned getID() {
#ifndef NDEBUG
    return id;
#else
    return 0;
#endif
  }
#ifndef NDEBUG
  void setID(unsigned i) { id = i; }
#endif
};

class ExprOperand {
  Expr *expr;

public:
  ExprOperand(Expr *) {}
  void set(Expr *newExpr) { expr = newExpr; }
  Expr *get() const { return expr; }
};

/// expressions in the same congruence class are all kept in thee same intrusive
/// list. The Congruence class stores this list.
/// expressions should never use the original in the hash or compare because
/// expressions can get cloned with a new original and need to compared equal
/// with there clone
class Expr : public llvm::ilist_node_with_parent<Expr, CongruenceClass>,
             public WithID {
public:
  enum ExprKind : unsigned {
    /// Default, similar to a structural hash and compare
    generic,

    /// Merge of control flow
    phi,

    /// Used for external values or as fallback
    external,

    constant,

    /// This represent an operations which as a GvnOpInterface
    custom,

    /// Represent an unreachable value, all dead expression are considered
    /// equals
    dead,
  };
  static StringRef exprKindToStr(ExprKind kind) {
    switch (kind) {
      // clang-format off
      case generic: return "generic";
      case phi: return "phi";
      case external: return "external";
      case constant: return "constant";
      case custom: return "custom";
      case dead: return "dead";
      // clang-format on
    }
    llvm_unreachable("unknown ExprKind");
  }

protected:
  /// The hash should be filled by the constructor of the derived class.
  /// Since nearly all expressions are put in a hash map. so we compute the hash
  /// eagerly.
  unsigned hash;

  /// The value in the original IR this expression represent.
  Value original;

  /// This is usually a Value but it can be an Attribute if the Expr represent a
  /// constant
  OpFoldResult current;
  ExprKind kind : 3;
  unsigned numOperands : 29;

  Expr(ExprKind k, Value orig, OpFoldResult curr,
       ArrayRef<Expr *> operands = {})
      : original(orig), current(curr), kind(k), numOperands(operands.size()) {
    initOperands(operands);
  }

  /// The space for operands is allocated before the Expr
  ExprOperand *getOperandsStart() {
    return reinterpret_cast<ExprOperand *>(this) - numOperands;
  }
  void initOperand(ExprOperand &op, Expr *expr) {
    /// initialize all the operands
    new (&op) ExprOperand(this);
    op.set(expr);
  }
  void initOperands(ArrayRef<Expr *> operands) {
    assert(operands.size() == numOperands);
    for (unsigned idx = 0; idx < operands.size(); idx++)
      initOperand(getOperands()[idx], operands[idx]);
  }

  bool isOperandEqual(Expr *other) {
    if (getOperands().size() != other->getOperands().size())
      return false;
    for (unsigned idx = 0; idx < getOperands().size(); idx++)
      if (getOperands()[idx].get()->getCurrent() !=
          other->getOperands()[idx].get()->getCurrent())
        return false;
    return true;
  }

  void copyFromImpl(Expr *other) {}

  unsigned computeOperandsHash() {
    auto hashRange = llvm::map_range(getOperands(), [](ExprOperand elem) {
      return elem.get()->getCurrent();
    });
    return llvm::hash_combine_range(hashRange.begin(), hashRange.end());
    return hash;
  }

public:
  void verifyInvariance() {
#ifndef NDEBUG
    assert(isEqual(this));
    assert(original);
    assert(current);
    dispatchToImpl(this, [](auto *expr) { expr->verifyInvarianceImpl(); });
#endif
  }
  struct EmptyTag {};
  Expr(EmptyTag) {}
  bool isInitial() { return !original; }
  Location getOrigLoc() { return original.getLoc(); }
  ExprKind getKind() const { return kind; }
  CongruenceClass *cClass = nullptr;
  MutableArrayRef<ExprOperand> getOperands() {
    return {getOperandsStart(), numOperands};
  }

  unsigned getCurrIdx() {
    if (auto currRes = dyn_cast_if_present<OpResult>(getCurrVal()))
      return currRes.getResultNumber();
    return 0;
  }
  CongruenceClass *getParent() { return cClass; }
  unsigned getHash() { return hash; }
  Value getOriginal() { return original; }
  PointerUnion<Attribute, Value> getCurrPointerUnion() {
    return *static_cast<PointerUnion<Attribute, Value> *>(&current);
  }
  OpFoldResult getCurrent() { return current; }
  Value getCurrVal() { return current.dyn_cast<Value>(); }
  Operation *getCurrOp() {
    if (auto val = getCurrVal())
      return val.getDefiningOp();
    return nullptr;
  }
  Attribute getCurrAttr() { return current.dyn_cast<Attribute>(); }
  bool isEqual(Expr *other);

  /// Used for cloning expression
  void copyFrom(Expr *other, Value original);
  void print(raw_ostream &os);
  void printAsValue(raw_ostream &os) { os << "Expr(" << getID() << ")"; }
  LLVM_DUMP_METHOD void dump() { return print(llvm::errs()); }

  /// Dispatch the lambda to the correct subclass of Expr
  template <typename RetTy = void, typename T = void>
  static RetTy dispatchToImpl(Expr *expr, T &&callable) {
    return llvm::TypeSwitch<Expr *, RetTy>(expr)
        .template Case<ExternalExpr, PHIExpr, GenericOpExpr, ConstExpr,
                       DeadExpr>(callable).getValue();
  }
};

/// Represent any generic operation(except constants) that can be merged to an
/// other expression with the same calculation
class GenericOpExpr : public Expr {
  /// Operands are ignored because we use Expr to for it
  /// This hash and compare are only intended to compare the action of the
  /// operation not if it processes the same data.
  /// Note: number of operands is checked with the operands
  static unsigned hashOpAction(Operation *op) {
    assert(op);
    return OperationEquivalence::computeHash(
        op,
        /*hashOperands=*/OperationEquivalence::ignoreHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isOpActionEqual(Operation *lhs, Operation *rhs) {
    assert(lhs && rhs);
    /// Same operation so they are the same
    if (lhs == rhs)
      return true;

    /// Structural comparaison
    return OperationEquivalence::isEquivalentTo(
        lhs, const_cast<Operation *>(rhs),
        /*mapOperands=*/OperationEquivalence::ignoreValueEquivalence,
        /*mapResults=*/OperationEquivalence::ignoreValueEquivalence,
        OperationEquivalence::IgnoreLocations);
  }

  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    return llvm::hash_combine(Expr::generic, hashOpAction(getCurrOp()),
                              getCurrIdx(), computeOperandsHash());
  }

public:
  GenericOpExpr(ArrayRef<Expr *> operands, Value orig, Value current)
      : Expr(Expr::generic, orig, current, operands) {
    hash = computeHash();
  }

  void verifyInvarianceImpl() {
    assert(hash == computeHash());
    Operation *op = cast<OpResult>(getCurrVal()).getOwner();
    assert(op->getNumOperands() == getOperands().size());
  }
  static bool classof(const Expr *expr) {
    return expr->getKind() == Expr::generic;
  }
  bool isEqual(GenericOpExpr *other) {
    if (!isOpActionEqual(getCurrOp(), other->getCurrOp()) ||
        getCurrIdx() != other->getCurrIdx())
      return false;
    return isOperandEqual(other);
  }
};

class CostumeExpr : public Expr {
  GvnOpInterface interface;
  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    return llvm::hash_combine(Expr::custom,
                              interface.computeOperationActionHash(
                                  getCurrOp(), computeOperandsHash()),
                              getCurrIdx());
  }

public:
  CostumeExpr(ArrayRef<Expr *> operands, Value orig, Value current, GvnOpInterface interface)
      : Expr(Expr::custom, orig, current, operands), interface(interface) {
    hash = computeHash();
  }

  void verifyInvarianceImpl() {
    assert(hash == computeHash());
    Operation *op = cast<OpResult>(getCurrVal()).getOwner();
    assert(op->getNumOperands() == getOperands().size());
  }
  static bool classof(const Expr *expr) {
    return expr->getKind() == Expr::generic;
  }
  bool isEqual(CostumeExpr *other) {
    if (!isOperandEqual(other))
      return false;
    SmallVector<PointerUnion<Attribute, Value>> lhsOperands(
        llvm::map_range(other->getOperands(), [](const ExprOperand &expr) {
          return expr.get()->getCurrPointerUnion();
        }));
    SmallVector<PointerUnion<Attribute, Value>> rhsOperands(
        llvm::map_range(getOperands(), [](const ExprOperand &expr) {
          return expr.get()->getCurrPointerUnion();
        }));
    return interface.isOperationActionEqual(getCurrOp(), other->getCurrOp(),
                                            lhsOperands, rhsOperands);
  }
};

/// An expression that cannot be matched by other. It is used for any thing we
/// cannot reason about. regions arguments, operations that access memory(since
/// we dont yet model it). since it is not congruent with any other Expr. it
/// doesn't need to track what the original operations depended upon.
class ExternalExpr : public Expr {
  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    return llvm::hash_combine(Expr::external, getCurrent());
  }

public:
  ExternalExpr(Value origAndCurrent)
      : Expr(Expr::external, origAndCurrent, origAndCurrent) {
    hash = computeHash();
  }
  static bool classof(const Expr *expr) {
    return expr->getKind() == Expr::external;
  }
  bool isEqual(ExternalExpr *other) {
    return getCurrent() == other->getCurrent();
  }
  void verifyInvarianceImpl() {
    assert(hash == computeHash());
    assert(getCurrVal());
  }
};

/// Represent a Value with an undef value. At the start every value is
/// represented by a DeadExpr.
class DeadExpr : public Expr {
  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    return llvm::hash_value(Expr::dead);
  }

public:
  DeadExpr(Value origAndCurrent)
      : Expr(Expr::dead, origAndCurrent, origAndCurrent) {
    hash = computeHash();
  }
  static bool classof(const Expr *expr) {
    return expr->getKind() == Expr::dead;
  }
  bool isEqual(DeadExpr *other) { return true; }
  void verifyInvarianceImpl() {
    assert(hash == computeHash());
  }
};

/// Represent a merge in control-flow. contrary to other ops the value
/// represented by a PHIExpr is a BlockArgument, no an OpResult.
class PHIExpr : public Expr {
  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    /// PHIs from different blocks may not have the same condition to split the
    /// value. So they cant be considered equal.
    return llvm::hash_combine(kind, getCurrVal().getParentBlock(),
                              computeOperandsHash());
  }

public:
  PHIExpr(ArrayRef<Expr *> operands, Value original)
      : Expr(Expr::phi, original, original, operands) {
    assert(operands.size() > 1 && "phi(x) should be x");
    initOperands(operands);
    hash = computeHash();
  }
  static bool classof(const Expr *expr) { return expr->getKind() == Expr::phi; }
  bool isEqual(PHIExpr *other) {
    return isOperandEqual(other) &&
           original.getParentBlock() == other->original.getParentBlock();
  }
  void verifyInvarianceImpl() {
    assert(hash == computeHash());
    assert(isa<BlockArgument>(original));
  }
};

/// Represent a constant. The constants are represented with attributes instead
/// of Values
class ConstExpr : public Expr {
  unsigned computeHash() {
    if (!isUsingProperHash)
      return 0;
    return llvm::hash_combine(kind, getCurrAttr());
  }

public:
  /// Keep track of the dialect such that the constant can get materialized
  Dialect *dialect;

  ConstExpr(Value orig, Attribute cst, Dialect *dialect)
      : Expr(Expr::constant, orig, cst), dialect(dialect) {
    hash = computeHash();
  }
  static bool classof(const Expr *expr) {
    return expr->getKind() == Expr::constant;
  }
  bool isEqual(ConstExpr *other) {
    /// Attribute are uniqued so if they are the same they have the same pointer
    return getCurrAttr() == other->getCurrAttr();
  }
  void copyFromImpl(ConstExpr *other) { dialect = other->dialect; }
  void verifyInvarianceImpl() {
    assert(hash == computeHash());
    assert(getCurrAttr());
  }
};

/// This is used to hash and compare expression that should be merged into the
/// same congruence class.
struct DenseMapExprUniquer : DenseMapInfo<Expr *> {
  static unsigned getHashValue(const Expr *val) {
    return const_cast<Expr *>(val)->getHash();
  }
  static bool isEqual(const Expr *lhsCst, const Expr *rhsCst) {
    Expr *lhs = const_cast<Expr *>(lhsCst);
    Expr *rhs = const_cast<Expr *>(rhsCst);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;

    /// in Hash maps usually only the low bits are taken in account for bucket
    /// finding. our hash is precomputed but our comparaison are expensive.
    if (lhs->getHash() != rhs->getHash())
      return false;

    return lhs->isEqual(rhs);
  }
};

bool Expr::isEqual(Expr *other) {
  /// Hash maps like DenseMap only use part of the hash for bucketing and not
  /// the full hash that is already calculated.
  if (getHash() != other->getHash())
    return false;

  Expr *lhs = this;
  Expr *rhs = other;

  if (lhs->kind != rhs->kind)
    return false;

  /// All expressions use different rules for how they can match each other. so
  /// delegate their implementation
  bool result = dispatchToImpl<bool>(
      lhs, [&](auto first) { return first->isEqual((decltype(first))rhs); });
  assert(!result ||
         lhs->getHash() == rhs->getHash() && "hash/comparaison mismatch");
  return result;
}

void Expr::copyFrom(Expr *other, Value orig) {
  numOperands = other->numOperands;
  kind = other->kind;
  hash = other->hash;
  current = other->current;
  original = orig;

  /// classes should be assigned by updateCongruenceFor and not here

  for (unsigned idx = 0; idx < other->getOperands().size(); idx++)
    initOperand(getOperands()[idx], other->getOperands()[idx].get());
  dispatchToImpl(
      this, [&](auto *expr) { expr->copyFromImpl((decltype(expr))other); });
}

/// Represent a set of Expr that are considered to be congruent.
/// all live CongruenceClass are kept in an intrusive list. that is pruned as
/// expressions are moved from one class to an other
struct CongruenceClass : public llvm::ilist_node<CongruenceClass>,
                         public WithID {
  using ilist = llvm::ilist_node<CongruenceClass>;
  ~CongruenceClass() { assert(members.empty()); }

  /// Should only be modified by addToClass
  /// Because the GVNState keeps track of which classes are live.
  /// And of whether or not the leader changed, when the leader changes all
  /// expressions in the class need to be reprocessed
  llvm::iplist<Expr> members;

  Expr *getLeader() {
    assert(!members.empty());
    return &members.front();
  }
  bool isInitial() { return isa<DeadExpr>(getLeader()); }

  LLVM_DUMP_METHOD void dump() { print(llvm::errs()); }
  void print(raw_ostream &os);
  static void printAsValue(raw_ostream &os, CongruenceClass *cClass) {
    os << "Class(";
    if (cClass)
      os << cClass->getID();
    else
      os << "null";
    os << ")";
  }
  void verifyInvariance() {
#ifndef NDEBUG
    assert(isInList() != members.empty());
    for (Expr &elem : members) {
      elem.verifyInvariance();
      assert(elem.cClass == this);
    }
#endif
  }
};

raw_ostream &operator<<(raw_ostream &os, Expr &expr) {
  expr.print(os);
  return os;
}

raw_ostream &operator<<(raw_ostream &os, CongruenceClass &expr) {
  expr.print(os);
  return os;
}

void CongruenceClass::print(raw_ostream &os) {
  os << "CongruenceClass ";
  os << "id=" << getID();
  os << " leader=";
  if (!members.empty())
    getLeader()->printAsValue(os);
  else
    os << "null";
  os << " members(" << std::distance(members.begin(), members.end()) << ")=\n";
  for (Expr &mem : members)
    os << mem << "\n";
}

void Expr::print(raw_ostream &os) {
  os << "Expr ";
  os << getID();
  os << " " << llvm::utohexstr(hash) << " ";
  os << " " << exprKindToStr(kind) << " ";
  CongruenceClass::printAsValue(os, cClass);

  auto printVal = [&](Value val) {
    if (!val) {
      os << "null";
      return;
    }
    os << llvm::utohexstr(((uintptr_t)val.getImpl()) & 0xffffff) << ":";
    if (BlockArgument arg = dyn_cast<BlockArgument>(val))
      os << "arg " << val.getType() << " at " << arg.getArgNumber();
    else
      os << val;
  };

  if (auto val = getCurrVal()) {
    os << " curr=";
    printVal(val);
  }
  if (auto attr = getCurrAttr())
    os << " curr=" << attr.getImpl() << ":\"" << attr << "\"";
  os << " orig=";
  printVal(getOriginal());
  for (ExprOperand &operand : getOperands()) {
    os << " ";
    operand.get()->printAsValue(os);
  }
}

} // namespace
namespace llvm {

template <>
struct DenseMapInfo<BlockEdge> {
  static inline BlockEdge getEmptyKey() { return {(Block *)1, nullptr}; }
  static inline BlockEdge getTombstoneKey() { return {nullptr, (Block *)1}; }
  static unsigned getHashValue(BlockEdge val) {
    return hash_combine(val.from, val.to);
  }
  static bool isEqual(BlockEdge lhs, BlockEdge rhs) {
    return lhs.from == rhs.from && lhs.to == rhs.to;
  }
};

} // namespace llvm

namespace {
namespace gvn {

struct GVNstate;

/// Global information about the current GVN pass.
/// It is mostly in charge of traversing the IR structure and run GVN on
/// regions.
struct GVNPass : public impl::GVNBase<GVNPass> {
  DominanceInfo *domInfo = nullptr;

  void processOp(Operation *op);
  void runOnOperation() override;
};

/// The factory for expressions and CongruenceClasses
class Allocator {
  unsigned exprID = 1;
  unsigned classID = 1;
  llvm::BumpPtrAllocator allocator;
  llvm::ArrayRecycler<Expr> recycler;
  using alloc_capacity = llvm::ArrayRecycler<Expr>::Capacity;

  void *allocate(size_t size) {
    return (void *)recycler.allocate(alloc_capacity::get(size), allocator);
  }
  template <typename T>
  void assertSimple() {
    static_assert(
        std::is_same_v<CongruenceClass, T> || std::is_same_v<ExternalExpr, T> ||
            std::is_same_v<DeadExpr, T> || std::is_same_v<ConstExpr, T>,
        "T must be simple");
  }
  template <typename T>
  void *allocComplex(unsigned count) {
    static_assert(alignof(T) == alignof(ExprOperand));
    unsigned operandSize = sizeof(ExprOperand) * count;
    char *startAddr = (char *)allocate(sizeof(T) + operandSize);
    return startAddr + operandSize;
  }

#ifndef NDEBUG
  Expr *assignID(Expr *expr) {
    assert(expr->getID() == 0);
    expr->setID(exprID++);
    exprs.push_back(expr);
    return expr;
  }
  CongruenceClass *assignID(CongruenceClass *cClass) {
    assert(cClass->getID() == 0);
    cClass->setID(classID++);
    classes.push_back(cClass);
    return cClass;
  }
  void remove(Expr *expr) {
    for (Expr *&elem : exprs)
      if (elem == expr)
        elem = nullptr;
  }
  void remove(CongruenceClass *cClass) {
    for (CongruenceClass *&elem : classes)
      if (elem == cClass)
        elem = nullptr;
  }
  SmallVector<Expr *> exprs;
  SmallVector<CongruenceClass *> classes;
#endif
public:
  template <typename T, typename... Ts>
  T *makeSimple(Ts &&...ts) {
    assertSimple<T>();
    auto *res = ::new (allocate(sizeof(T))) T(std::forward<Ts>(ts)...);
#ifndef NDEBUG
    assignID(res);
#endif
    return res;
  }
  template <typename T>
  void deleteImpl(size_t allocSize, T *ptr) {
#ifndef NDEBUG
    remove(ptr);
#endif
    ptr->~T();
    recycler.deallocate(alloc_capacity::get(allocSize), (Expr *)ptr);
  }
  template <typename T, typename... Ts>
  T *makeComplex(ArrayRef<Expr *> operands, Ts &&...ts) {
    T *res = new (allocComplex<T>(operands.size())) T(operands, ts...);
#ifndef NDEBUG
    assignID(res);
#endif
    return res;
  }
  void deleteObj(CongruenceClass *ptr) {
    deleteImpl(sizeof(CongruenceClass), ptr);
  }
  void deleteObj(Expr *ptr) {
    deleteImpl(Expr::dispatchToImpl<unsigned>(
                   ptr,
                   [](auto *ptr) {
                     return sizeof(std::remove_pointer<decltype(ptr)>);
                   }) +
                   ptr->getOperands().size() * sizeof(ExprOperand),
               ptr);
  }
  Expr *cloneExpr(Expr *from, Value original) {
    void *addr = Expr::dispatchToImpl<void *>(from, [&](auto *expr) {
      return allocComplex<std::remove_pointer_t<decltype(expr)>>(
          from->getOperands().size());
    });
    Expr *res = new (addr) Expr(Expr::EmptyTag{});
#ifndef NDEBUG
    assignID(res);
#endif
    res->copyFrom(from, original);
    return res;
  }
  LLVM_DUMP_METHOD void dump();

  /// This is pretty expensive
  void verifyInvariance() {
#ifndef NDEBUG
    for (CongruenceClass *c : classes)
      if (c)
        c->verifyInvariance();
    for (Expr *e : exprs)
      if (e)
        e->verifyInvariance();
#endif
  }
  ~Allocator() { recycler.clear(allocator); }
};

/// Numbering of a block
struct NumRange {
  /// first argument of the block
  unsigned begin = 0;
  /// first operation of the block and 1 past last argument of the block
  unsigned argEnd = 0;
  /// one past the last operation of the block
  unsigned end = 0;
  bool isInvalid() { return begin == 0 || argEnd == 0 || end == 0; }
};

/// Encapsulate all the numbering an update tracking system
/// It mostly exist to add asserts on every edits to the system.
/// And to make it easy to find how a value got updated.
/// By creating a global place to place breakpoints
class UpdateAndNumberingTracker {
  /// Each Block argument and operation has a bit in this vector for wether or
  /// not it needs to get updated
  BitVector touchedValues;
  DenseMap<ValOrOp, unsigned> valOrOpToNum;
  SmallVector<ValOrOp, 32> numToValOrOp;
  DenseMap<Block *, NumRange> blockOpRange;

public:
  /// Used to setup the tracker
  class Builder {
    UpdateAndNumberingTracker &impl;
    NumRange currentBlock;
    unsigned valCounter = 1;

  public:
    unsigned blockCounter = 1;
    unsigned edgeCounter = 0;
    Builder(UpdateAndNumberingTracker &i) : impl(i) {
      impl.numToValOrOp.emplace_back(nullptr);
    }
    void startBlock() { currentBlock.begin = valCounter; }
    void endArgs() { currentBlock.argEnd = valCounter; }
    void endBlock(Block *b) {
      currentBlock.end = valCounter;
      impl.blockOpRange[b] = currentBlock;
      currentBlock = NumRange{};
    }
    void reserveBlocks(unsigned count) { impl.blockOpRange.reserve(count); }
    void assignNumbering(ValOrOp v) {
      LLVM_DEBUG(llvm::dbgs() << "num=" << valCounter << " " << v << "\n");
      impl.valOrOpToNum[v] = valCounter++;
      impl.numToValOrOp.emplace_back(v);
    };
    void finalize() { impl.touchedValues.resize(valCounter); }
  };
  unsigned lookupNumOr0(ValOrOp val) {
    assert(!val.isNull());
    return valOrOpToNum.lookup(val);
  }
  unsigned lookupNum(ValOrOp val) {
    unsigned res = lookupNumOr0(val);
    assert(res);
    return res;
  }
  ValOrOp lookupVal(unsigned num) {
    assert(num);
    ValOrOp res = numToValOrOp[num];
    assert(!res.isNull());
    return res;
  }
  NumRange lookupRange(Block *b) {
    NumRange r = blockOpRange.lookup(b);
    assert(!r.isInvalid());
    return r;
  }
  void set(unsigned idx) { touchedValues.set(idx); }
  void set(ValOrOp val) { set(lookupNum(val)); }
  void set() { touchedValues.set(); }
  void reset(unsigned idx) { touchedValues.reset(idx); }
  void reset(ValOrOp val) { reset(lookupNum(val)); }
  void set(unsigned begin, unsigned end) {
    assert(begin != 0 && end != 0);
    touchedValues.set(begin, end);
  }
  void reset(unsigned begin, unsigned end) {
    assert(begin != 0 && end != 0);
    touchedValues.reset(begin, end);
  }
  const BitVector &getTouchedIndexes() { return touchedValues; }
  LLVM_DUMP_METHOD void dump() {
    llvm::errs() << "touched ops and vals(" << touchedValues.count() << "):\n";
    for (unsigned num : touchedValues.set_bits()) {
      ValOrOp val = lookupVal(num);
      llvm::errs() << "num=" << num << " " << val << "\n";
    }
  }
  bool isBackEdge(BlockEdge edge) {
    return edge.from == edge.to ||
           /// We can argEnd or end instead of begin, all leads to the same
           /// result
           blockOpRange[edge.from].begin > blockOpRange[edge.to].begin;
  }
  unsigned getRank(Value v) {
    Operation *op = v.getDefiningOp();
    if (!op)
      return valOrOpToNum[v];
    unsigned num = valOrOpToNum[op];
    assert(num != 0);
    num |= cast<OpResult>(v).getResultNumber() << 24;
    return num;
  }
};

/// Local information about the GVN pass. there is one of these per operation
/// isolatedFromAbove
struct GVNstate {
  GVNstate(GVNPass &);
  GVNPass &global;

  Allocator alloc;
  UpdateAndNumberingTracker tracker;

  /// NewGVN calls it Top
  llvm::iplist<CongruenceClass> liveClasses;
  DenseMap<Value, Expr *> valueToExpr;
  void removeFromClass(Expr *expr) {
    /// remove expr from the class
    expr->cClass->members.remove(expr);
    /// Cleanup the class if it is now empty
    if (expr->cClass->members.empty()) {
      liveClasses.remove(expr->cClass);
      alloc.deleteObj(expr->cClass);
    }
  }
  Expr *lookupExpr(Value val) {
    Expr *expr = valueToExpr.lookup(val);
    assert(expr);
    return expr;
  }
  void updateExpr(Expr *expr, Value val) {
    assert(expr->getOriginal() == val);
    Expr *&current = valueToExpr[val];
    assert(current->getOriginal() == val);
    removeFromClass(current);

    /// expressions can still be used by other expression that dont need an
    /// update. so we do not delete them
    current = expr;
  }
  /// Add or transfer v to newClass
  /// Also keeps liveClasses up to date
  void addToClass(Expr *expr, CongruenceClass *newClass) {
    /// If the newClass is not yet live make it live
    if (!newClass->isInList())
      liveClasses.push_back(newClass);

    /// If expr already has a class
    if (expr->cClass)
      removeFromClass(expr);
    /// add expr to the new class
    newClass->members.push_back(expr);
    expr->cClass = newClass;
    exprMerger[expr] = expr->cClass;
  }

  /// Expr* is not hash and compared based on its pointer but based on the
  /// content of the Expr
  DenseMap<Expr *, CongruenceClass *, DenseMapExprUniquer> exprMerger;
  Expr *lookupLeader(Value val) {
    Expr *expr = lookupExpr(val);
    if (expr->cClass) {
      if (isa<DeadExpr>(expr->cClass->getLeader()))
        /// TODO: This should return undef
        return expr;
      return expr->cClass->getLeader();
    }
    return expr;
  }

  /// Reachability tacking
  llvm::SmallDenseSet<Block *, 1> reachableBlocks;
  DenseSet<BlockEdge> reachableEdges;

  SmallVector<Operation *> temporaries;
  /// Values coming from outside of the basic regions.
  /// These values cannot be analyzed
  /// The same Value may be used more then once but we want to build it only
  /// once. Maybe we could a use a vector here since the only change would be
  /// the order of a few debug prints.
  llvm::SmallSetVector<Value, 4> externalValues;

  Region *region = nullptr;
  DominanceInfo *getDom();

  LLVM_DUMP_METHOD void dump();

  /// verification
  void verifyReachedFixpoint();

  /// Initialization
  void setupFor(Region &r);
  void initCongruenceClasses();

  /// Iteration
  void updateCongruenceFor(Expr *expr, Value val);
  void processGenericOp(Operation *op, SmallVectorImpl<Expr *> &res);
  void updateReachableEdge(BlockEdge edge);
  void processTerminator(Operation *op, SmallVectorImpl<Expr *> &res);
  void processBlockArg(BlockArgument arg, SmallVectorImpl<Expr *> &res);
  void processValOrOp(ValOrOp iterable, SmallVectorImpl<Expr *> &res);
  void iterate();

  /// Change the IR
  Value getConstantFor(ConstExpr* cstExpr);
  void performChanges();
  void cleanup();

  void run();
};

LLVM_DUMP_METHOD void Allocator::dump() {
  llvm::errs() << "classes (" << classes.size() << "):\n";
  for (CongruenceClass *cClass : classes)
    if (cClass) {
      cClass->dump();
      llvm::errs() << "\n";
    }
  llvm::errs() << "exprs (" << exprs.size() << "):\n";
  for (Expr *elem : exprs)
    if (elem) {
      elem->dump();
      llvm::errs() << "\n";
    }
}

LLVM_DUMP_METHOD void GVNstate::dump() {
  AsmState asmState(region->getParentOp());
  struct PrinterHook final : PrinterHookBase {
    GVNstate *state;
    void printCommentBeforeOp(Operation *op, raw_ostream &os,
                              unsigned currentIndent) override {
      if (op->getParentRegion() != state->region)
        return;
      for (OpResult val : op->getResults()) {
        os.indent(currentIndent);
        os << "// res " << val.getResultNumber() << " ";
        state->lookupExpr(val)->print(os);
        os << "\n";
      }
    }
    void printCommentBeforeBlock(Block *block, raw_ostream &os,
                                 unsigned currentIndent) override {}
  } printHook;
  printHook.state = this;
  region->getParentOp()->print(llvm::errs(), asmState, &printHook);
}

void GVNstate::initCongruenceClasses() {
  /// Create the default class. This GVN is optimistic so everything left inside
  /// it at the end is dead or undef
  MLIRContext *ctx = region->getContext();
  IRRewriter rewriter(ctx);
    CongruenceClass *initialClass = alloc.makeSimple<CongruenceClass>();

  /// Go through every reachable blocks
  for (Block &b : region->getBlocks()) {
    if (!getDom()->isReachableFromEntry(&b))
      continue;

    /// Add every Value to the initialClass
    auto add = [&](Value val) {
      Expr *expr = valueToExpr[val] = alloc.makeSimple<DeadExpr>(val);
      addToClass(expr, initialClass);
    };

    /// the entry block doesn't have block arguments because there is no edges
    /// coming into it.
    if (&b != &region->front())
      for (BlockArgument &arg : b.getArguments())
        add(arg);
    for (Operation &o : b.getOperations())
      for (Value res : o.getResults())
        add(res);
  }

  /// Arguments of the region are all unique so they have their own class
  for (BlockArgument &arg : region->front().getArguments()) {
    Expr *expr = alloc.makeSimple<ExternalExpr>(arg);
    valueToExpr[arg] = expr;
    addToClass(expr, alloc.makeSimple<CongruenceClass>());
    LLVM_DEBUG(llvm::dbgs() << "arg expr:" << *expr << "\n");
  }

  /// We cant analyses values coming from outside the region. so we all put them
  /// into external class
  for (Value val : externalValues) {
    Expr *expr = alloc.makeSimple<ExternalExpr>(val);
    valueToExpr[val] = expr;
    addToClass(expr, alloc.makeSimple<CongruenceClass>());
    LLVM_DEBUG(llvm::dbgs() << "external operand:" << *expr << "\n");
  }
}

void GVNstate::updateCongruenceFor(Expr *expr, Value val) {
  CongruenceClass *currentClass = lookupExpr(val)->cClass;
  Expr* currentExpr = lookupExpr(val);
  if (currentExpr->isEqual(expr)) {
    /// Expr has just been built so we have the only reference to it at this
    /// point. But as soon as it is placed in the map, other expression will start
    /// depending on it.
    alloc.deleteObj(expr);
    return;
  }
  auto lookupResult = exprMerger.insert({expr, nullptr});

  /// There is no match so we create a new congruence class
  if (lookupResult.second)
    /// Update the map
    lookupResult.first->second = alloc.makeSimple<CongruenceClass>();
  CongruenceClass *newClass = lookupResult.first->second;

  /// If the class was the same we should have bailed earlier
  assert(currentClass != newClass);

  /// If expr is the current leader of the old class.
  /// For update the old class members
  if (currentClass && currentClass->getLeader() == expr)
    for (Expr &elem : currentClass->members)
      tracker.set(elem.getOriginal());

  /// Move expr to the new class
  addToClass(expr, newClass);

  /// expr is in a new class, update its users
  for (OpOperand operand : val.getUsers())
    if (operand.getOwner()->getParentRegion() == region)
      /// Unreachable code is not in tracked but can still use our values
      /// So we skip if the lookup returns 0, 0 means not tracked so unreachable
      /// This will also skip Uses outside of the current region
      if (unsigned num = tracker.lookupNumOr0(operand.getOwner()))
        tracker.set(num);
  updateExpr(expr, val);

  LLVM_DEBUG(llvm::dbgs() << "updated:" << *newClass);
}

void GVNstate::processBlockArg(BlockArgument arg,
                               SmallVectorImpl<Expr *> &res) {
  /// Block arguments are processed like we would process a phi node.

  /// operands are uniqued by there Expr*, operands are kept in order. because
  /// phi(a, b) != phi(b, a)
  SetVector<Expr *, SmallVector<Expr *, 4>,
            llvm::SmallDenseSet<Expr *, 4, DenseMapExprUniquer>>
      exprs;
  Block *curr = arg.getParentBlock();
  for (auto it = curr->pred_begin(); it != curr->pred_end(); it++) {
    Block *pred = *it;

    /// If the edge is unreachable, skip it.
    if (!reachableEdges.contains({pred, curr}))
      continue;
    auto br = cast<BranchOpInterface>(pred->getTerminator());
    Value operand =
        br->getOperand(br.getSuccessorOperands(it.getSuccessorIndex())
                           .getOperandIndex(arg.getArgNumber()));
    Expr *expr = lookupExpr(operand);

    /// initial is undef so: phi(undef, ...) -> phi(...)
    if (expr->cClass->isInitial())
      continue;

    exprs.insert(expr);
  }

  /// TODO: add more phi folding technics

  if (exprs.empty()) {
    res.push_back(alloc.makeSimple<DeadExpr>(arg));
    return;
  }

  /// phi(a) == a
  if (exprs.size() == 1) {
    res.push_back(alloc.cloneExpr(exprs.front(), arg));
    return;
  }

  res.push_back(alloc.makeComplex<PHIExpr>(exprs.getArrayRef(), arg));
}

void GVNstate::updateReachableEdge(BlockEdge edge) {
  NumRange range = tracker.lookupRange(edge.to);
  if (reachableEdges.insert(edge).second)
    if (reachableBlocks.insert(edge.to).second) {
      /// This block has never been visited so process all block args and ops in
      /// the block
      tracker.set(range.begin, range.end);
      return;
    }
  /// This block has already been visited only reprocess block arguments
  tracker.set(range.begin, range.argEnd);
}

void GVNstate::processTerminator(Operation *op, SmallVectorImpl<Expr *> &res) {
  assert(op->hasTrait<OpTrait::IsTerminator>());

  /// In case the terminator has some results
  processGenericOp(op, res);

  /// Try to constant fold the branch based on current knowledge
  if (auto br = dyn_cast<BranchOpInterface>(op)) {
    SmallVector<Attribute> currentConstants;

    /// Try to get constants for every operands
    for (Value operand : op->getOperands())
      if (auto* cstExpr = dyn_cast<ConstExpr>(lookupExpr(operand)))
        currentConstants.push_back(cstExpr->getCurrAttr());
      else
        currentConstants.push_back(nullptr);

    /// If the branch gets constant folded. only update reachability the edge
    /// that got used
    if (Block *successor = br.getSuccessorForOperands(currentConstants)) {
      updateReachableEdge({op->getBlock(), successor});
      return;
    }
  }

  /// If constant folding failed, update reachability of every edge.
  for (Block *to : op->getSuccessors())
    updateReachableEdge({op->getBlock(), to});
}

void GVNstate::processGenericOp(Operation *op, SmallVectorImpl<Expr *> &res) {
  /// fallback for now
  if (!MemoryEffectOpInterface::hasNoEffect(op)) {
    for (Value val : op->getResults())
      res.push_back(alloc.makeSimple<ExternalExpr>(val));
    return;
  }

  SmallVector<Value, 4> inputs(op->getOperands());
  if (op->hasTrait<OpTrait::IsCommutative>())
    llvm::sort(inputs, [&](Value lhs, Value rhs) {
      int lhsID = tracker.getRank(lhs);
      int rhsID = tracker.getRank(rhs);
      assert(lhs == rhs || lhsID != rhsID);
      return lhsID < rhsID;
    });

  SmallVector<Attribute, 4> constInput;
  SmallVector<Expr *, 4> exprs;

  for (Value &in : inputs) {
    Expr *leader = lookupLeader(in);
    exprs.push_back(leader);
    Attribute constant = leader->getCurrAttr();
    in = leader->getCurrVal() ? leader->getCurrVal() : in;
    constInput.push_back(constant);
  }

  /// Dont try to fold if it has a region
  if (!op->getNumRegions()) {

    Operation *newOp = op->clone();
    /// Update operands with known information from the GVN
    newOp->setOperands(inputs);

    SmallVector<OpFoldResult> foldResult;
    if (succeeded(newOp->fold(constInput, foldResult))) {
      if (foldResult.empty()) {
        /// Folded to an updated operation
        assert(false && "this path is still un tested");
        LLVM_DEBUG(llvm::dbgs()
                   << "folded: \"" << *op << "\" to: \"" << newOp << "\"\n");
        temporaries.push_back(newOp);
        return processGenericOp(newOp, res);
      }
      /// Folded to a constant or external, so we dont need to keep the
      /// operation around
      LLVM_DEBUG(llvm::dbgs() << "folded: \"" << *op << "\" to:";
                 for (OpFoldResult val
                      : foldResult) llvm::dbgs()
                 << " \"" << val << "\"";
                 llvm::dbgs() << "\n");

      newOp->erase();
      for (unsigned idx = 0; idx < foldResult.size(); idx++) {
        if (Value val = foldResult[idx].dyn_cast<Value>())
          res.push_back(
              alloc.cloneExpr(lookupExpr(val), op->getResults()[idx]));
        else
          res.push_back(alloc.makeSimple<ConstExpr>(
              op->getResult(idx), foldResult[idx].dyn_cast<Attribute>(),
              op->getDialect()));
      }
      return;
    }
    newOp->erase();
  }

  for (OpResult v : op->getResults()) {
    if (auto gvnInterface = dyn_cast<GvnOpInterface>(op))
      res.push_back(alloc.makeComplex<CostumeExpr>(exprs, v, v, gvnInterface));
    res.push_back(alloc.makeComplex<GenericOpExpr>(exprs, v, v));
  }
}

void GVNstate::processValOrOp(ValOrOp iterable, SmallVectorImpl<Expr *> &res) {
  /// Dispatch the ValOrOp
  if (auto arg = dyn_cast_if_present<BlockArgument>(iterable.dyn_cast<Value>()))
    return processBlockArg(arg, res);
  Operation *op = iterable.dyn_cast<Operation *>();
  if (op->hasTrait<OpTrait::IsTerminator>())
    return processTerminator(op, res);
  return processGenericOp(op, res);
}

void GVNstate::verifyReachedFixpoint() {
#ifndef NDEBUG
  LLVM_DEBUG(llvm::dbgs() << "verifying a fixedpoint was reached:\n");
  /// To verify that we reached a fixpoint:
  ///  - copy the current mappings
  ///  - touch everything
  ///  - rerun the iteration, with all the current information
  ///  - verify that nothing moved.

  auto valueToExprCopy = valueToExpr;
  /// congruence class of expressions also need to be tracked
  DenseMap<Expr*, CongruenceClass*> exprToClassCopy;
  for (auto valExpr : valueToExprCopy)
    exprToClassCopy[valExpr.second] = valExpr.second->cClass;

  tracker.set();
  /// 0 doesn't represent a value, it is used for error detection.
  tracker.reset(0);
  /// Also dont process block arguments of the entry block
  NumRange idxRange = tracker.lookupRange(&region->front());
  tracker.reset(idxRange.begin, idxRange.argEnd);

  iterate();
  for (auto valExpr : valueToExprCopy) {
    Value val = valExpr.first;
    Expr* oldExpr = valExpr.second;
    Expr* newExpr = valueToExpr.lookup(val);

    /// New expressions are always created so check for structural equality
    /// updateCongruenceFor should not replace the expression if it is
    /// equivalent to the current expression so we can compare pointers here.
    assert(oldExpr == newExpr);
    assert(oldExpr->cClass == newExpr->cClass);
  }
#endif
}

void GVNstate::iterate() {
  uint64_t iterations = 0;

  Block *lastBlock = &region->front();

  /// While we have work to do
  while (tracker.getTouchedIndexes().any()) {
    LLVM_DEBUG(llvm::dbgs() << "iteration " << iterations << "\n");

    /// This might happens naturally but it is much more likely that it is an
    /// infinite loop.
    assert(iterations < 20);
    ++iterations;

    /// Go thought every ops we need to process
    for (unsigned valNum : tracker.getTouchedIndexes().set_bits()) {
      tracker.reset(valNum);
      ValOrOp iterable = tracker.lookupVal(valNum);
      Block *currentBlock = getBlock(iterable);

      /// We changed block
      if (lastBlock != currentBlock) {
        lastBlock = currentBlock;

        /// If the block is not reach we should not process it
        if (!reachableBlocks.count(currentBlock)) {
          NumRange r = tracker.lookupRange(currentBlock);
          tracker.reset(r.begin, r.end);
          continue;
        }
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "processing: num=" << valNum << " " << iterable << "\n");
      SmallVector<Expr *, 4> results;
      processValOrOp(iterable, results);
      if (auto val = iterable.dyn_cast<Value>()) {
        assert(results.size() == 1);
        updateCongruenceFor(results[0], val);
      } else {
        Operation *op = iterable.get<Operation *>();
        assert(op->getNumResults() == results.size());
        for (unsigned idx = 0; idx < op->getNumResults(); idx++)
          updateCongruenceFor(results[idx], op->getResult(idx));
      }
      alloc.verifyInvariance();
    }
  }
}

Value GVNstate::getConstantFor(ConstExpr *cstExpr) {
  MLIRContext *ctx = region->getContext();
  IRRewriter rewriter(ctx);
  OperationFolder folder(ctx);
  rewriter.setInsertionPointToStart(&region->front());
  Value result;

  auto *origOp = cstExpr->getOriginal().getDefiningOp();
  if (origOp && origOp->hasTrait<OpTrait::ConstantLike>()) {
    origOp->moveBefore(&region->front().front());
    result = cstExpr->getOriginal();
  } else {
    result = folder.getOrCreateConstant(
        rewriter, cstExpr->dialect, cstExpr->getCurrAttr(),
        cstExpr->getOriginal().getType(), cstExpr->getOrigLoc());
    assert(result && "failed to materialize constant");
  }
  return result;
}

void GVNstate::performChanges() {
  MLIRContext *ctx = region->getContext();
  IRRewriter rewriter(ctx);
  OperationFolder folder(ctx);
  rewriter.setInsertionPointToStart(&region->front());
  llvm::SmallPtrSet<Operation *, 16> tryCleanupOp;
  llvm::SmallPtrSet<BlockArgument, 16> cleanupBlockArg;

  for (CongruenceClass &cClass : liveClasses) {
    /// TODO: replace initial by undef
    if (cClass.isInitial())
      continue;

    Value replacement = cClass.getLeader()->getCurrVal();
    if (ConstExpr *cst = dyn_cast<ConstExpr>(cClass.getLeader()))
      replacement = getConstantFor(cst);

    for (Expr &elem : cClass.members) {
      /// dont replace the leader
      if (elem.getOriginal() == replacement)
        continue;

      /// If the replacement doesn't dominate every use, skip it.
      /// TODO: sinking to users
      if (!llvm::all_of(elem.getOriginal().getUsers(), [&](Operation *user) {
            return getDom()->dominates(replacement, user);
          }))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "replacing: " << elem.getOriginal() << " by "
                              << replacement << "\n");
      elem.getOriginal().replaceAllUsesWith(replacement);

      /// If we removed all use of a block argument, erase it.
      if (auto arg = dyn_cast<BlockArgument>(elem.getOriginal()))
        if (!arg.getParentBlock()->isEntryBlock())
          cleanupBlockArg.insert(arg);

      if (auto *op = elem.getOriginal().getDefiningOp())
        tryCleanupOp.insert(op);
    }
  }

  for (Operation *op : tryCleanupOp)
    if (isOpTriviallyDead(op)) {
      LLVM_DEBUG(llvm::dbgs() << "erasing: " << *op << "\n");
      rewriter.eraseOp(op);
    }

  for (BlockArgument arg : cleanupBlockArg) {
    assert(arg.use_empty());
    LLVM_DEBUG(llvm::dbgs() << "erasing: " << arg << "\n");
    int idx = arg.getArgNumber();
    Block *curr = arg.getOwner();
    for (auto it = curr->pred_begin(); it != curr->pred_end(); it++) {
      Block *pred = *it;

      auto br = cast<BranchOpInterface>(pred->getTerminator());
      br.getSuccessorOperands(it.getSuccessorIndex()).erase(idx);
    }
    curr->eraseArgument(idx);
  }
}

void GVNstate::cleanup() {
  /// everything is allocated inside a BumpAllocator that is soon going out of
  /// scope. so nothing is going to leak. and there is no need to spend time
  /// unlinking everything
  liveClasses.clearAndLeakNodesUnsafely();
}

void GVNstate::run() {
  if (!region)
    return;

  /// The GVN has already been setup for its region at this point
  initCongruenceClasses();
  iterate();
  verifyReachedFixpoint();
  performChanges();

  cleanup();
}

void GVNstate::setupFor(Region &r) {
  UpdateAndNumberingTracker::Builder builder(tracker);
  /// Block arguments are used to present PHI nodes.

  auto checkForExternalValues = [&](Operation &op) {
    /// Values with defs outside of the regions need special handling so we
    /// list them here.
    for (Value val : op.getOperands())
      if (val.getParentRegion() != &r)
        externalValues.insert(val);
  };

  /// For some reason Dominator tree doest work for block with only one block
  /// Se here is our fallback
  if (r.hasOneBlock()) {
    /// BlockArgument are not used for control-flow updates but they are needed
    /// to make sure every value has a different ID when ordering in commutative
    /// ops
    builder.startBlock();
    /// This is not needed or control-flow but just for value ranking
    for (BlockArgument &arg : r.front().getArguments())
      builder.assignNumbering(arg);
    builder.endArgs();
    for (Operation &op : r.front().getOperations()) {
      checkForExternalValues(op);
      builder.assignNumbering(&op);
    }
    builder.endBlock(&r.front());
  } else {
    llvm::SmallDenseMap<Block *, unsigned, 16> rpoOrdering;
    llvm::ReversePostOrderTraversal<Region *> rpot(&r);
    for (auto &b : rpot) {
      rpoOrdering[b] = builder.blockCounter++;
      builder.edgeCounter += b->getNumSuccessors();
    }
    // Sort dominator tree children arrays into RPO.
    for (auto &b : rpot) {
      auto *node = getDom()->getNode(b);
      if (node->getNumChildren() > 1)
        llvm::sort(*node, [&](DominanceInfoNode *lhs, DominanceInfoNode *rhs) {
          return rpoOrdering[lhs->getBlock()] < rpoOrdering[rhs->getBlock()];
        });
    }

    builder.reserveBlocks(builder.blockCounter);
    llvm::df_iterator_default_set<DominanceInfoNode *, 16> reachable;
    for (auto *domNode :
         depth_first_ext(getDom()->getRootNode(&r), reachable)) {
      Block *b = domNode->getBlock();

      /// Generate a numbering for a Value or an Op that we may need to process
      builder.startBlock();
      for (BlockArgument &arg : b->getArguments())
        builder.assignNumbering(arg);
      builder.endArgs();
      for (Operation &op : b->getOperations()) {
        checkForExternalValues(op);
        builder.assignNumbering(&op);
      }

      builder.endBlock(b);
    }
  }

  /// Prepare the size of containers
  reachableBlocks.reserve(builder.blockCounter);
  reachableEdges.reserve(builder.edgeCounter);

  /// Mark entry block as live
  reachableBlocks.insert(&r.front());

  assert(!region && "only support one region for now");
  region = &r;

  builder.finalize();
  NumRange range = tracker.lookupRange(&region->front());

  /// Touch all operation results not block arguments
  tracker.set(range.argEnd, range.end);
}

DominanceInfo *GVNstate::getDom() { return global.domInfo; }
GVNstate::GVNstate(GVNPass &g) : global(g) {}

void GVNPass::processOp(Operation *op) {
  for (Region &r : op->getRegions()) {
    {
      GVNstate state(*this);
      state.setupFor(r);
      state.run();
    }
    for (Block &b : r.getBlocks())
      for (Operation &o : b.getOperations())
        processOp(&o);
  }
}

void GVNPass::runOnOperation() {
  isUsingProperHash = hashCollisions;
  domInfo = &getAnalysis<DominanceInfo>();
  processOp(getOperation());
}

} // namespace gvn
} // namespace

namespace mlir {

std::unique_ptr<Pass> createGVNPass() {
  return std::make_unique<gvn::GVNPass>();
}

} // namespace mlir
