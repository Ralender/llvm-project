//===- ObfuscateCF.cpp - Obfuscate Control-Flow
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Turn non-conditional branches into conditional branches with only one side
// used and obfuscate the condition to make it harder to figure out that half
// the branch is dead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ObfuscateCF.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"

#include <random>

#define DEBUG_TYPE "obfuscate-cf"

using namespace llvm;

/// This exist to make the behavior deterministic.
cl::opt<uint64_t> Seed("obfuscate-cf-seed",
                       cl::desc("Seed for random number generator for CF "
                                "obfuscation, 0 mean random (default = 1)"),
                       cl::init(1));

uint64_t getSeed() {
  if (Seed == 0)
    return std::random_device()();
  return Seed;
}

struct ObfuscateCFState {
  Function &F;
  IRBuilder<> Builder;
  DominatorTree &DT;

  /// Set of all block with an incoming back-edge.
  DenseSet<BasicBlock *> ReceiveBackedge;

  /// Collection of values that can be used for obfuscation purposes
  DenseMap<Type *, SmallVector<Value *>> ValueMap;
  std::mt19937 Rng;

  ObfuscateCFState(Function &F, DominatorTree &DT)
      : F(F), Builder(F.getContext()), DT(DT), Rng(getSeed()) {}

  int getRand(int max) {
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, max);
    return dist(Rng);
  }

  /// Check if V dominates The current inserting point of the builder
  /// Sadly this is not already part of the DT API
  bool dominateBuilder(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      /// if I is inside the same block as the insert point
      if (I->getParent() == Builder.GetInsertBlock()) {
        /// if I is the end insertion point, it dominateBuilder the whole block
        if (Builder.GetInsertPoint() == Builder.GetInsertBlock()->end())
          return true;
        /// else if depends what comes first
        return I->comesBefore(&*Builder.GetInsertPoint());
      }
      return DT.dominates(I, Builder.GetInsertBlock());
    }
    /// Otherwise it is argument, global or constant all of which dominate
    /// everything in the function
    return true;
  }

  /// Get a random value of type Ty from the function, that dominates the
  /// current insertion point of the builder
  Value *getAnyValueOfType(Type *Ty) {
    if (ValueMap.count(Ty)) {
      auto &Values = ValueMap[Ty];
      llvm::shuffle(Values.begin(), Values.end(), Rng);
      for (Value *V : Values)
        if (dominateBuilder(V))
          return V;
    }
    /// TODO: Lookup Values of other types and cast them
    return Constant::getNullValue(Ty);
  }

  Value *getObfuscatedFalse() {
    /// for any A, popcount(A) < popcount(A & (A - 1)) is false
    /// neither of GCC or clang seem to be able to break this down.
    /// This function will generate the above expression with some value in the
    /// function
    Value *A = getAnyValueOfType(Builder.getInt32Ty());
    Value *ASub1 = Builder.CreateSub(A, Builder.getInt32(1));
    Value *AAndASub1 = Builder.CreateAnd(A, ASub1);
    Function *PopCountFunc = Intrinsic::getDeclaration(
        F.getParent(), Intrinsic::ctpop, {Builder.getInt32Ty()});
    Value *PopCountR = Builder.CreateCall(PopCountFunc, {AAndASub1});
    Value *PopCountL = Builder.CreateCall(PopCountFunc, {A});
    return Builder.CreateICmpULT(PopCountL, PopCountR);
    /// TODO: add more ways to obfuscate the condition
  }

  void collectValue(Value *V) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": collecting " << *V << "\n");
    ValueMap[V->getType()].push_back(V);
  }

  BasicBlock *getBlockFor(Value *V) {
    if (auto *I = dyn_cast<Instruction>(V))
      return I->getParent();
    return nullptr;
  }

  /// This is called after a terminator has been moved from From to To.
  /// It will updated PHI nodes in the successor of To.
  void updatePHIs(BasicBlock *From, BasicBlock *To) {
    for (unsigned i = 0; i < To->getTerminator()->getNumSuccessors(); i++) {
      BasicBlock *BB = To->getTerminator()->getSuccessor(i);
      for (PHINode &PHI : BB->phis()) {
        int idx = PHI.getBasicBlockIndex(From);
        if (idx != -1) {
          PHI.setIncomingBlock(idx, To);
        }
      }
    }
  }

  void collectBackedges() {
    /// Simple DFS to find add blocks receiving backedges in
    /// ReceiveBackedge
    BasicBlock *BB = &F.getEntryBlock();
    if (succ_empty(BB))
      return;

    SmallPtrSet<const BasicBlock *, 8> Visited;
    SmallVector<std::pair<const BasicBlock *, unsigned>, 8> VisitStack;
    SmallPtrSet<const BasicBlock *, 8> InStack;

    Visited.insert(BB);
    VisitStack.push_back(std::make_pair(BB, 0));
    InStack.insert(BB);
    do {
      std::pair<const BasicBlock *, unsigned> &Top = VisitStack.back();
      const BasicBlock *ParentBB = Top.first;
      unsigned &I = Top.second;

      bool FoundNew = false;
      while (I != ParentBB->getTerminator()->getNumSuccessors()) {
        BB = ParentBB->getTerminator()->getSuccessor(I);
        if (Visited.insert(BB).second) {
          FoundNew = true;
          break;
        }
        // Successor is in VisitStack, it's a back edge.
        if (InStack.count(BB))
          ReceiveBackedge.insert(BB);
        I++;
      }

      if (FoundNew) {
        InStack.insert(BB);
        VisitStack.push_back(std::make_pair(BB, 0));
      } else {
        InStack.erase(VisitStack.pop_back_val().first);
      }
    } while (!VisitStack.empty());
  }

  /// Core logic to turn an non-conditional branch into a conditional branch
  void obfuscateBR(BranchInst *BR) {
    /// Nothing to do if the branch is conditional
    if (BR->isConditional())
      return;
    BasicBlock *To = BR->getSuccessor(0);
    BasicBlock *From = BR->getParent();

    /// Don't obfuscate blocks with incoming back-edges to keep it simple.
    if (ReceiveBackedge.count(To))
      return;

    /// If the block has no successors because it terminates with a ret or
    /// unreachable, split it. to get a successor
    if (To->getTerminator()->getNumSuccessors() == 0) {
      To = To->splitBasicBlock(To->getTerminator(), "", /*Before*/ true);
    }
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": processing " << *BR << "\n");

    /// Update the CFG:
    /// Before:
    ///       \|/
    ///      From  ...
    ///        \  /
    ///         To
    ///        /|\
    /// After:
    ///        \|/
    ///       From  ...
    ///       /  \  /
    ///ExtraBB    To
    ///       \  /
    ///      JoinBB
    ///        /|\
    ///

    /// Create and insert the new blocks
    BasicBlock *ExtraBB = BasicBlock::Create(F.getContext());
    BasicBlock *JoinBB = BasicBlock::Create(F.getContext());
    F.insert(F.end(), ExtraBB);
    F.insert(F.end(), JoinBB);

    /// Replace the non-conditional branch with a conditional branch.
    Instruction *OldTerm = From->getTerminator();
    Builder.SetInsertPoint(From);
    Builder.CreateCondBr(getObfuscatedFalse(), ExtraBB, To);
    OldTerm->eraseFromParent();

    /// Link ExtraBB to JoinBB
    Builder.SetInsertPoint(ExtraBB);
    Builder.CreateBr(JoinBB);

    /// Move the terminator of the To block to the JoinBB. The condition not
    /// dominating its new use will be fixed below
    Builder.SetInsertPoint(JoinBB);
    Instruction *Term = To->getTerminator();
    Term->removeFromParent();
    Builder.Insert(Term);

    /// Update PHI nodes in blocks that used to be successors of To and are now
    /// successors of JoinBB
    updatePHIs(To, JoinBB);

    /// Link To to JoinBB
    Builder.SetInsertPoint(To);
    Builder.CreateBr(JoinBB);

    /// TODO: partial updates of the dominator tree instead of full recalculations
    /// Recalculate the dominator tree after all CFG changes have been made
    DT.recalculate(F);

    /// Update the use-def chain:
    /// PHIs will be inserted into the JoinBB and stores will be inserted into ExtraBB
    for (Instruction &I : *To) {
      if (I.isTerminator())
        continue;
      if (auto *SI = dyn_cast<StoreInst>(&I))
        /// TODO: If the pointer operand cannot be used. it could be recreated
        /// on the ExtraBB
        if (DT.dominates(SI->getPointerOperand(), &ExtraBB->front())) {
          /// Create stores the same address as the original block if it is
          /// trivially possible
          Builder.SetInsertPoint(ExtraBB, ExtraBB->getFirstInsertionPt());
          Builder.CreateStore(
              getAnyValueOfType(SI->getValueOperand()->getType()),
              SI->getPointerOperand());
        }

      /// If the Value is used outside of the To block.
      if (any_of(I.users(), [&](User *U) { return getBlockFor(U) != To; })) {
        /// Build a PHI to merge it with a value coming from ExtraBB
        Builder.SetInsertPoint(JoinBB, JoinBB->getFirstInsertionPt());
        PHINode *PHI = Builder.CreatePHI(I.getType(), 2);

        /// The value form the original block
        PHI->addIncoming(&I, To);

        /// Use a random value of the function since this code is dead anyway.
        PHI->addIncoming(getAnyValueOfType(I.getType()), ExtraBB);

        /// Replace cross-blocks uses of I with the new PHI
        I.replaceUsesWithIf(PHI, [&](Use &U) {
          return getBlockFor(U.getUser()) != To && U.getUser() != PHI;
        });
      }
    }
    /// TODO: more random stuff could be inserted in the ExtraBB since it will
    /// never be used

    LLVM_DEBUG(verifyFunction(F));
    return;
  }

  void run() {
    /// Arguments may be used as replacements in obfuscating code
    for (Value &Arg : F.args()) {
      collectValue(&Arg);
    }

    collectBackedges();
    /// TODO: artificially insert unconditional branches to create even more
    /// messy control-flow

    /// Collect basic blocks
    SmallVector<BasicBlock *> BBs;
    for (BasicBlock &BB : F)
      BBs.push_back(&BB);

    for (BasicBlock *BB : BBs) {
      if (auto *BR = dyn_cast<BranchInst>(BB->getTerminator()))
        obfuscateBR(BR);
      /// Collect values of the block for use in obfuscating code
      for (Instruction &I : *BB)
        collectValue(&I);
    }
  }
};

PreservedAnalyses ObfuscateCFPass::run(Function &F,
                                       FunctionAnalysisManager &FAM) {
  ObfuscateCFState{F, FAM.getResult<DominatorTreeAnalysis>(F)}.run();
  return PreservedAnalyses::none();
}
