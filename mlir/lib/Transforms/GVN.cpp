
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/TypeID.h"
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
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
#define GEN_PASS_DEF_GVN
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mlir-gvn"

using namespace mlir;

namespace {

struct BlockEdge {
  Block *from;
  Block *to;
};

using ValOrOp = llvm::PointerUnion<Value, Operation *>;

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
struct VariableExpr;
struct ShallowExpr;
struct PHIExpr;
struct GenericOpExpr;
struct ConstExpr;
struct DeadExpr;

class WithID {
  unsigned id = 0;
  public:
  unsigned getID() const { return id; }
  void setID(unsigned i) { id = i; }
};

struct ExprOperand : IROperand<ExprOperand, Expr *, Expr *> {
  using base = IROperand<ExprOperand, Expr *, Expr *>;
  using base::base;
  static IRObjectWithUseList<ExprOperand, Expr *> *getUseList(Expr *value);
};

/// Found by ADL
llvm::hash_code hash_value(const ExprOperand &value) {
  return llvm::hash_value(value.get());
}

class Expr : public IRObjectWithUseList<ExprOperand, Expr *>,
             public llvm::ilist_node_with_parent<Expr, CongruenceClass>,
             public WithID {
public:
  enum ExprKind : unsigned {
    /// Default, similar to a structural hash and compare
    generic,

    /// Merge of control flow
    phi,

    /// An expression with the same hash a comparaison as another. inner
    /// Expression
    shallow,

    /// Used for external values or as fallback
    variable,

    constant,

    /// Represent an unreachable value, all dead expression are considered
    /// equals
    dead,
  };
  static StringRef exprKindToStr(ExprKind kind) {
    switch (kind) {
      // clang-format off
      case generic: return "generic";
      case phi: return "phi";
      case shallow: return "shallow";
      case variable: return "variable";
      case constant: return "constant";
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
  const OpFoldResult current;
  const ExprKind kind : 3;
  const unsigned numOperands : 29;

  Expr(ExprKind k, Value orig, OpFoldResult curr,
       ArrayRef<Expr *> operands = {})
      : original(orig), current(curr), kind(k), numOperands(operands.size()) {}

  /// The space for operands is allocated before the Expr
  ExprOperand *getOperandsStart() {
    return reinterpret_cast<ExprOperand *>(this) - numOperands;
  }
  ExprOperand *getOperandsStart() const {
    return const_cast<Expr *>(this)->getOperandsStart();
  }
  void setOperands(ArrayRef<Expr *> operands) {
    assert(operands.size() == numOperands);
    for (unsigned idx = 0; idx < operands.size(); idx++) {
      /// initialize all the operands
      new (&getOperands()[idx]) ExprOperand(this);
      getOperands()[idx].set(operands[idx]);
    }
  }

  bool isOperandEqual(const Expr *other) const {
    if (getOperands().size() != other->getOperands().size())
      return false;
    for (unsigned idx = 0; idx < getOperands().size(); idx++)
      if (getOperands()[idx].get() != other->getOperands()[idx].get())
        return false;
    return true;
  }

public:
  bool isInitial() const { return !original; }
  Location getOrigLoc() const { return original.getLoc(); }
  ExprKind getKind() const { return kind; }
  CongruenceClass *cClass = nullptr;
  MutableArrayRef<ExprOperand> getOperands() {
    return {getOperandsStart(), numOperands};
  }
  ArrayRef<ExprOperand> getOperands() const {
    return {getOperandsStart(), numOperands};
  }

  unsigned getCurrIdx() const {
    if (auto currRes = dyn_cast_if_present<OpResult>(getCurrVal()))
      return currRes.getResultNumber();
    return 0;
  }
  CongruenceClass *getParent() const { return cClass; }
  unsigned getHash() const { return hash; }
  Value getOriginal() const { return original; }
  OpFoldResult getCurrent() const { return current; }
  Value getCurrVal() const { return current.dyn_cast<Value>(); }
  Operation *getCurrOp() const {
    if (auto val = getCurrVal())
      return val.getDefiningOp();
    return nullptr;
  }
  Attribute getCurrAttr() const { return current.dyn_cast<Attribute>(); }
  bool isEqual(const Expr *other) const;
  void print(raw_ostream &os) const;
  void printAsValue(raw_ostream &os) const {
    os << "Expr(" << getID() << ")";
  }
  LLVM_DUMP_METHOD void dump() const { return print(llvm::errs()); }
  /// Dispatch the lambda to the correct subclass of Expr
  template <typename RetTy = void, typename T = void>
  static RetTy dispatchToImpl(Expr *e, T &&callable) {
    return llvm::TypeSwitch<Expr *, RetTy>(e)
        .template Case<VariableExpr, ShallowExpr, PHIExpr, GenericOpExpr,
                       ConstExpr>(callable);
  }
};

/// Represent any generic operation, but not constants
struct GenericOpExpr : Expr {
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
    return llvm::hash_combine(kind, hashOpAction(getCurrOp()), getCurrIdx(),
                              getOperands());
  }

  GenericOpExpr(ArrayRef<Expr *> operands, Value orig, Value current)
      : Expr(Expr::generic, orig, current, operands) {
    hash = computeHash();
  }
public:
  static bool classof(const Expr *e) { return e->getKind() == Expr::generic; }
  bool isEqual(const GenericOpExpr *other) {
    if (!isOpActionEqual(getCurrOp(), other->getCurrOp()) ||
        getCurrIdx() != other->getCurrIdx())
      return false;
    return isOperandEqual(other);
  }
};

/// An expression that no other value with match with
struct VariableExpr : Expr {
  VariableExpr(Value origAndCurrent)
      : Expr(Expr::variable, origAndCurrent, origAndCurrent) {
    hash = llvm::DenseMapInfo<Value>::getHashValue(getOriginal());
  }
  static bool classof(const Expr *e) { return e->getKind() == Expr::variable; }
  bool isEqual(const VariableExpr *other) { return getOriginal() == other->getOriginal(); }
};

struct DeadExpr : Expr {
  DeadExpr(Value origAndCurrent)
      : Expr(Expr::dead, origAndCurrent, origAndCurrent) {
    hash = llvm::hash_value(Expr::dead);
  }
  static bool classof(const Expr *e) { return e->getKind() == Expr::dead; }
  bool isEqual(const DeadExpr *other) { return true; }
};

struct ShallowExpr : Expr {
  static Expr *stripShallow(const Expr *exprCst) {
    Expr *expr = const_cast<Expr *>(exprCst);
    auto *maybeShallow = dyn_cast<ShallowExpr>(expr);
    if (!maybeShallow)
      return expr;
    assert(!isa<ShallowExpr>(maybeShallow->inner));
    return maybeShallow->inner;
  }
  Expr *inner;
  ShallowExpr(ArrayRef<Expr *> operands, Expr *other, Value original)
      : Expr(Expr::shallow, original, other->getCurrent(), operands),
        inner(stripShallow(other)) {
    setOperands(operands);
    hash = inner->getHash();
  }
  static bool classof(const Expr *e) { return e->getKind() == Expr::shallow; }
  bool isEqual(const Expr *other) {
    llvm_unreachable("should never be called");
  }
};

struct PHIExpr : Expr {
  PHIExpr(ArrayRef<Expr *> operands, Value original)
      : Expr(Expr::phi, original, nullptr, operands) {
    assert(operands.size() > 1 && "should be a shallow");
    setOperands(operands);
    hash = llvm::hash_combine(kind, original.getParentBlock(), getOperands());
  }
  static bool classof(const Expr *e) { return e->getKind() == Expr::phi; }
  bool isEqual(const PHIExpr *other) {
    return isOperandEqual(other) &&
           original.getParentBlock() == other->original.getParentBlock();
  }
};

struct ConstExpr : Expr {
  Dialect* dialect;
  ConstExpr(Value orig, Attribute cst, Dialect *dialect)
      : Expr(Expr::constant, orig, cst), dialect(dialect) {
    hash = llvm::hash_combine(kind, getCurrAttr());
  }
  static bool classof(const Expr *e) { return e->getKind() == Expr::constant; }
  bool isEqual(const ConstExpr *other) {
    /// Attribute are uniqued so if they are the same they have the same pointer
    return getCurrAttr() == other->getCurrAttr();
  }
};

IRObjectWithUseList<ExprOperand, Expr *> *ExprOperand::getUseList(Expr *value) {
  return value;
}

/// This is used to hash and compare expression that should be merged into the
/// same congruence class
struct DenseMapExprUniquer : DenseMapInfo<Expr *> {
  static unsigned getHashValue(Expr *val) { return val->getHash(); }
  static bool isEqual(const Expr *lhs, const Expr *rhs) {
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

bool Expr::isEqual(const Expr *other) const {
  if (getHash() != other->getHash())
    return false;
  Expr *lhs = ShallowExpr::stripShallow(this);
  Expr *rhs = ShallowExpr::stripShallow(other);

  if (lhs->kind != rhs->kind)
    return false;
  bool result = dispatchToImpl<bool>(
      lhs, [&](auto first) { return first->isEqual((decltype(first))rhs); });
  assert(!result ||
         lhs->getHash() == rhs->getHash() && "hash/comparaison mismatch");
  return result;
}

struct CongruenceClass : public llvm::ilist_node<CongruenceClass>,
                         public WithID {
  using ilist = llvm::ilist_node<CongruenceClass>;
  CongruenceClass(Value l) : leader(l) {}

  /// TODO: make sure it is always dominating every other member
  Value leader;

  /// Should only be modified by addToClass
  /// Because the GVNState keeps track of which classes are live
  llvm::iplist<Expr> members;

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
  os << " leader=" << leader << " members("
     << std::distance(members.begin(), members.end()) << ")=\n";
  for (Expr &mem : members)
    os << mem << "\n";
}

void Expr::print(raw_ostream &os) const {
  os << "Expr{";
  os << getID();
  os << " " << llvm::utohexstr(hash);
  os << " " << exprKindToStr(kind) << " ";
  CongruenceClass::printAsValue(os, cClass);

  /// For VariableExpr current == orig so no need to print both.
  /// For for shallow current == inner->current
  if (!isa<ShallowExpr, VariableExpr>(this)) {
    if (auto val = getCurrVal())
      os << " curr=" << val.getImpl() << ":\"" << val << "\"";
    if (auto attr = getCurrAttr())
      os << " curr=" << attr.getImpl() << ":\"" << attr << "\"";
  }
  os << " orig=" << getOriginal().getImpl() << ":\"" << getOriginal() << "\" ";
  if (auto *se = dyn_cast<ShallowExpr>(this)) {
    os << " inner=";
    se->inner->printAsValue(os);
  }
  for (const ExprOperand &operand : getOperands()) {
    os << " ";
    operand.get()->printAsValue(os);
  }
  os << "}";
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
struct GVNPass : public impl::GVNBase<GVNPass> {
  DominanceInfo *domInfo = nullptr;

  GVNstate *s = nullptr;
  GVNstate &getState() { return *s; }

  LogicalResult processOp(Operation *op);
  void runOnOperation() override;
};

class Allocator {
  unsigned exprID = 1;
  unsigned classID = 1;
  llvm::BumpPtrAllocator allocator;
  llvm::ArrayRecycler<Expr> recycler;
  using alloc_capacity = llvm::ArrayRecycler<Expr>::Capacity;
  public:
  void *allocate(size_t size) {
    return (void *)recycler.allocate(alloc_capacity::get(size), allocator);
  }
  template <typename T>
  void assertSimple() {
    static_assert(
        std::is_same_v<CongruenceClass, T> || std::is_same_v<VariableExpr, T> ||
            std::is_same_v<DeadExpr, T> || std::is_same_v<ConstExpr, T>,
        "T must be simple");
  }
  template <typename T, typename... Ts>
  T *makeSimple(Ts &&...ts) {
    assertSimple<T>();
    auto *res = ::new (allocate(sizeof(T))) T(std::forward<Ts>(ts)...);
    assignID(res);
    return res;
  }
  template <typename T>
  void deleteImpl(size_t allocSize, T *ptr) {
    ptr->~T();
    recycler.deallocate(alloc_capacity::get(allocSize), (Expr *)ptr);
  }
  template <typename T, typename... Ts>
  T *makeComplex(ArrayRef<Expr *> operands, Ts &&...ts) {
    static_assert(alignof(T) == alignof(ExprOperand));
    unsigned operandSize = sizeof(ExprOperand) * operands.size();
    char *startAddr = (char *)allocate(sizeof(T) + operandSize);
    void *objAddr = startAddr + operandSize;
    T *res = new (objAddr) T(operands, ts...);
    assignID(res);
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
  Expr *assignID(Expr *expr) {
    assert(expr->getID() == 0);
    expr->setID(exprID++);
    return expr;
  }
  CongruenceClass *assignID(CongruenceClass *cClass) {
    assert(cClass->getID() == 0);
    cClass->setID(classID++);
    return cClass;
  }
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
  unsigned lookupNum(ValOrOp val) {
    assert(!val.isNull());
    unsigned res = valOrOpToNum.lookup(val);
    assert(res);
    return res;
  }
  ValOrOp lookupVal(unsigned num) const {
    assert(num);
    ValOrOp res = numToValOrOp[num];
    assert(!res.isNull());
    return res;
  }
  NumRange lookupRange(Block *b) const {
    NumRange r = blockOpRange.lookup(b);
    assert(!r.isInvalid());
    return r;
  }
  void set(unsigned idx) { touchedValues.set(idx); }
  void set(ValOrOp val) { set(lookupNum(val)); }
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
  const BitVector &getTouchedIndexes() const { return touchedValues; }
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

struct ExprVal {
  Expr* expr;
  Value val;
};

struct ExprValMapInfo {
  static ExprVal getEmptyKey() {
    return {DenseMapInfo<Expr *>::getEmptyKey(), nullptr};
  }
  static ExprVal getTombstoneKey() {
    return {DenseMapInfo<Expr *>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(ExprVal val) {
    return DenseMapExprUniquer::getHashValue(val.expr);
  }
  static bool isEqual(ExprVal lhs, ExprVal rhs) {
    return DenseMapExprUniquer::isEqual(lhs.expr, rhs.expr);
  }
};

/// Local information about the GVN pass. there is one of these per operation
/// isolatedFromAbove
struct GVNstate {
  GVNstate(GVNPass &);
  GVNPass &global;

  Allocator alloc;
  UpdateAndNumberingTracker tracker;

  DenseMap<Value, Expr *> valueToExpr;
  /// Expr* is not hash and compared based on its pointer but based on the
  /// content of the Expr
  DenseMap<Expr *, CongruenceClass *, DenseMapExprUniquer> exprMerger;
  Value lookupLeader(Value val) {
    Expr *expr = valueToExpr.lookup(val);
    if (expr->cClass) {
      if (expr->cClass == initialClass)
        /// TODO: This should return poison
        return val;
      return expr->cClass->leader;
    }
    return val;
  }

  /// NewGVN calls it Top
  CongruenceClass *initialClass;
  llvm::iplist<CongruenceClass> liveClasses;
  /// Add or transfer v to newClass
  /// Also keeps liveClasses up to date
  void addToClass(Expr *expr, CongruenceClass *newClass) {
    /// If the newClass is not yet live make it live
    if (!newClass->isInList())
      liveClasses.push_back(newClass);

    /// If expr already has a class
    if (expr->cClass) {
      /// remove expr from the class
      expr->cClass->members.remove(expr);
      /// Cleanup the class if it is now empty
      if (expr->cClass->members.empty()) {
        liveClasses.remove(expr->cClass);
        alloc.deleteObj(expr->cClass);
      }
    }
    /// add expr to the new class
    newClass->members.push_back(expr);
    expr->cClass = newClass;
    assert(exprMerger.count(expr));
    exprMerger[expr] = expr->cClass;
  }

  /// Reachability tacking
  llvm::SmallDenseSet<Block *, 1> reachableBlocks;
  DenseSet<BlockEdge> reachableEdges;
  SmallVector<Operation*> temporaries;

  Region *region = nullptr;
  DominanceInfo *getDom();

  /// Initialization
  LogicalResult isValidOp(Region &r, Operation*);
  LogicalResult setupFor(Region &r);
  void initCongruenceClasses();

  /// Iteration
  void updateCongruenceFor(ExprVal ev);
  void processGenericOp(Operation *op, SmallVectorImpl<Expr*> &res);
  void updateReachableEdge(BlockEdge edge);
  void processTerminator(Operation *op, SmallVectorImpl<Expr*> &res);
  void processBlockArg(BlockArgument arg, SmallVectorImpl<Expr*> &res);
  void processValOrOp(ValOrOp iterable, SmallVectorImpl<Expr*> &res);
  void iterate();

  /// Change the IR
  void performChanges();
  void cleanup();

  void run();
};

/// RAII Scope to swap GVNstate
class Scope {
  GVNstate s;
  llvm::SaveAndRestore<GVNstate *> restore;

public:
  Scope(GVNPass &g) : s(g), restore(g.s, &s) {}
  ~Scope() {
    assert(s.global.s == &s);
    s.run();
  }
};

void GVNstate::initCongruenceClasses() {
  /// Create the default class. This GVN is optimistic so everything left inside
  /// it at the end is dead or poison
  MLIRContext *ctx = region->getContext();
  IRRewriter rewriter(ctx);
  initialClass = alloc.makeSimple<CongruenceClass>(nullptr);

  /// Go through every reachable blocks
  for (Block &b : region->getBlocks()) {
    if (!getDom()->isReachableFromEntry(&b))
      continue;

    /// Add every Value to the initialClass
    auto add = [&](Value val) {
      Expr *expr = valueToExpr[val] = alloc.makeSimple<DeadExpr>(val);
      addToClass(expr, initialClass);
    };

    for (BlockArgument &arg : b.getArguments())
      add(arg);
    for (Operation &o : b.getOperations())
      for (Value res : o.getResults())
        add(res);
  }

  /// Arguments of the region are all unique so they have their own class
  for (BlockArgument &arg : region->front().getArguments()) {
    Expr *e = alloc.makeSimple<VariableExpr>(arg);
    valueToExpr[arg] = e;
    addToClass(e, alloc.makeSimple<CongruenceClass>(arg));
    LLVM_DEBUG(llvm::dbgs() << "arg expr:" << *e << "\n");
  }
}

void GVNstate::updateCongruenceFor(ExprVal ev) {
  Expr* expr = ev.expr;
  Value val = ev.val;
  CongruenceClass *currentClass = expr->cClass;
  auto lookupResult = exprMerger.insert({expr, nullptr});

  /// There is no match so we create a new congruence class
  if (lookupResult.second)
    /// Update the map
    lookupResult.first->second = alloc.makeSimple<CongruenceClass>(val);
  CongruenceClass *newClass = lookupResult.first->second;
  if (currentClass != newClass) {
    addToClass(expr, newClass);

    /// The class has changed so update every user
    for (OpOperand operand : val.getUsers())
      if (operand.getOwner()->getParentRegion() == region)
        tracker.set(operand.getOwner());
  }
  LLVM_DEBUG(llvm::dbgs() << "updated class:" << *newClass);
  valueToExpr[val] = expr;
}

void GVNstate::processBlockArg(BlockArgument arg,
                               SmallVectorImpl<Expr*> &res) {
  /// Block arguments are processed like we would process a phi node.

  /// operands are uniqued by there Expr*
  SetVector<Expr *, SmallVector<Expr *, 4>,
            llvm::SmallDenseSet<Expr *, 4, DenseMapExprUniquer>>
      exprs;
  Block *curr = arg.getParentBlock();
  for (auto it = curr->pred_begin(); it != curr->pred_end(); it++) {
    Block* pred = *it;

    /// If the edge is unreachable, skip it.
    if (!reachableEdges.contains({pred, curr}))
      continue;
    auto br = cast<BranchOpInterface>(pred->getTerminator());
    Value operand =
        br->getOperand(br.getSuccessorOperands(it.getSuccessorIndex())
                           .getOperandIndex(arg.getArgNumber()));
    Expr *expr = valueToExpr.lookup(operand);

    /// initial is poison so: phi(poison, x) -> x
    if (expr->cClass == initialClass)
      continue;

    exprs.insert(expr);
  }

  /// TODO: add more phi folding technics

  /// phi of a single Expr is that expression
  if (exprs.size() == 1) {
    res.push_back(alloc.makeComplex<ShallowExpr>(
        ArrayRef<Expr *>{exprs.front()}, exprs.front(), arg));
    return;
  }

  res.push_back(alloc.makeComplex<PHIExpr>(exprs.getArrayRef(), arg));
}

void GVNstate::updateReachableEdge(BlockEdge edge) {
  if (reachableEdges.insert(edge).second) {
    NumRange range = tracker.lookupRange(edge.to);
    if (reachableBlocks.insert(edge.to).second)
      /// This block has never been visited so process all block args and ops in
      /// the block
      tracker.set(range.begin, range.end);
    else
      /// This block has already been visited only reprocess block arguments
      tracker.set(range.begin, range.argEnd);
  }
}

void GVNstate::processTerminator(Operation *op, SmallVectorImpl<Expr*> &res) {
  assert(op->hasTrait<OpTrait::IsTerminator>());

  /// Update reachability, and touched instructions
  for (Block *to : op->getSuccessors())
    updateReachableEdge({op->getBlock(), to});

  /// In case the terminator has some results
  processGenericOp(op, res);
}

void GVNstate::processGenericOp(Operation *op, SmallVectorImpl<Expr*> &res) {
  /// fallback for now
  if (!MemoryEffectOpInterface::hasNoEffect(op)) {
    for (Value val : op->getResults())
      res.push_back(alloc.makeSimple<VariableExpr>(val));
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

  /// Dont try to fold if it has a region
  if (!op->getNumRegions()) {
    SmallVector<Value, 4> inputLeaders;
    SmallVector<Attribute, 4> constInput;

    /// We assume that the input is already canonical, so if we have no more
    /// information than the op previously had we will not try to fold
    bool shouldTryTyFold = false;
    for (Value in : inputs) {
      Value leader = lookupLeader(in);
      shouldTryTyFold |= (in != leader);
      Attribute constant;
      if (matchPattern(leader, m_Constant(&constant)))
        shouldTryTyFold |= 1;
      inputLeaders.push_back(leader);
      constInput.push_back(constant);
    }
    if (shouldTryTyFold || op->hasTrait<OpTrait::ConstantLike>()) {
      Operation *newOp = op->clone();
      /// Update operands with known information from the GVN
      newOp->setOperands(inputLeaders);

      SmallVector<OpFoldResult> foldResult;
      if (succeeded(newOp->fold(constInput, foldResult))) {
        if (foldResult.empty()) {
          /// Folded to an updated operation
          LLVM_DEBUG(llvm::dbgs() << "folded: \"" << *op << "\" to: \"" << newOp << "\"\n");
          temporaries.push_back(newOp);
          return processGenericOp(newOp, res);
        }
        /// Folded to a constant or variable
        LLVM_DEBUG(llvm::dbgs() << "folded: \"" << *op << "\" to:";
                   for (OpFoldResult val
                        : foldResult) llvm::dbgs()
                   << " \"" << val << "\"";
                   llvm::dbgs() << "\n");
        newOp->erase();
        for (unsigned idx = 0; idx < foldResult.size(); idx++) {
          if (Value val = foldResult[idx].dyn_cast<Value>())
            res.push_back(
                alloc.makeComplex<ShallowExpr>({}, valueToExpr[val], inputs[idx]));
          else
            res.push_back(alloc.makeSimple<ConstExpr>(
                op->getResults()[idx], foldResult[idx].dyn_cast<Attribute>(),
                op->getDialect()));
        }
        return;
      }
      newOp->erase();
    }
  }

  SmallVector<Expr *, 4> exprs;
  llvm::transform(inputs, std::back_inserter(exprs),
                  [&](Value val) { return valueToExpr[val]; });

  for (OpResult v : op->getResults())
    res.push_back(alloc.makeComplex<GenericOpExpr>(exprs, v, v));
}

void GVNstate::processValOrOp(ValOrOp iterable, SmallVectorImpl<Expr*> &res) {
  /// Dispatch the ValOrOp
  if (auto arg = dyn_cast_if_present<BlockArgument>(iterable.dyn_cast<Value>()))
    return processBlockArg(arg, res);
  Operation *op = iterable.dyn_cast<Operation *>();
  if (op->hasTrait<OpTrait::IsTerminator>())
    return processTerminator(op, res);
  return processGenericOp(op, res);
}

void GVNstate::iterate() {
  uint64_t iterations = 0;

  Block *lastBlock = &region->front();

  /// While we have work to do
  while (tracker.getTouchedIndexes().any()) {
    LLVM_DEBUG(llvm::dbgs() << "iteration " << iterations << "\n");
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
        updateCongruenceFor({results[0], val});
      } else {
        Operation *op = iterable.get<Operation *>();
        assert(op->getNumResults() == results.size());
        for (unsigned idx = 0; idx < op->getNumResults(); idx++)
          updateCongruenceFor({results[idx], op->getResult(idx)});
      }
    }
  }
}

void GVNstate::performChanges() {
  MLIRContext *ctx = region->getContext();
  IRRewriter rewriter(ctx);
  OperationFolder folder(ctx);
  rewriter.setInsertionPointToStart(&region->front());
  llvm::SmallPtrSet<Operation *, 16> tryCleanupOp;
  llvm::SmallPtrSet<BlockArgument, 16> cleanupBlockArg;

  for (CongruenceClass &cClass : liveClasses) {
    /// TODO: replace initial by poison
    if (!cClass.leader)
      continue;

    for (Expr &e : cClass.members) {
      Value val = e.getCurrent().dyn_cast<Value>();
      if (ConstExpr *cst = dyn_cast<ConstExpr>(&e)) {
        val = folder.getOrCreateConstant(
            rewriter, cst->dialect, cst->getCurrAttr(),
            cst->getOriginal().getType(), cst->getOrigLoc());
        assert(val && "failed to materialize constant");
      }
      /// TODO: make constant
      if (val == cClass.leader)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "replacing: " << val << " by "
                              << cClass.leader << "\n");
      val.replaceAllUsesWith(cClass.leader);

      /// If we removed all use of a block argument, erase it.
      if (auto arg = dyn_cast<BlockArgument>(val))
        if (!arg.getParentBlock()->isEntryBlock())
          cleanupBlockArg.insert(arg);

      if (auto *op = val.getDefiningOp())
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
    Block* curr = arg.getOwner();
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
  performChanges();

  cleanup();
}

LogicalResult GVNstate::isValidOp(Region &r, Operation *o) {
  for (Value val : o->getOperands()) {
    if (val.getParentRegion() != &r) {
      o->emitError("def outside of region");
      return failure();
    }
    for (OpOperand &use : val.getUses()) {
      if (use.getOwner()->getParentRegion() != &r) {
        o->emitError("use outside of region");
        return failure();
      }
    }
  }
  return success();
}

LogicalResult GVNstate::setupFor(Region &r) {
  UpdateAndNumberingTracker::Builder builder(tracker);

  /// For some reason Dominator tree doest work for block with only one block
  /// Se here is our fallback
  if (r.hasOneBlock()) {
    /// BlockArgument are not used for control-flow updates but they are needed
    /// to make sure every value has a different ID when ordering in commutative
    /// ops
    builder.startBlock();
    for (BlockArgument &arg : r.front().getArguments())
      builder.assignNumbering(arg);
    builder.endArgs();
    for (Operation &op : r.front().getOperations()) {
      if (failed(isValidOp(r, &op)))
        return failure();
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
      builder.startBlock();
      /// Generate a numbering for a Value or an Op that we may need to process
      for (BlockArgument &arg : b->getArguments())
        builder.assignNumbering(arg);
      builder.endArgs();
      for (Operation &o : b->getOperations()) {
        if (failed(isValidOp(r, &o)))
          return failure();
        builder.assignNumbering(&o);
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
  return success();
}

DominanceInfo *GVNstate::getDom() { return global.domInfo; }
GVNstate::GVNstate(GVNPass &g) : global(g) {}

LogicalResult GVNPass::processOp(Operation *op) {
  for (Region &r : op->getRegions()) {
    {
      Scope scope(*this);
      if (failed(getState().setupFor(r)))
        return failure();
    }
    for (Block &b : r.getBlocks())
      for (Operation &o : b.getOperations())
        if (failed(processOp(&o)))
          return failure();
  }
  return success();
}

void GVNPass::runOnOperation() {
  domInfo = &getAnalysis<DominanceInfo>();
  if (failed(processOp(getOperation())))
    signalPassFailure();
}

} // namespace gvn
} // namespace

namespace mlir {

std::unique_ptr<Pass> createGVNPass() {
  return std::make_unique<gvn::GVNPass>();
}

} // namespace mlir
