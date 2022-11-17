//===- ViewLikeInterface.td - ViewLike interface -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for view-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GVN_OP_INTERFACE
#define MLIR_INTERFACES_GVN_OP_INTERFACE

include "mlir/IR/OpBase.td"

def GvnOpInterface : OpInterface<"GvnOpInterface"> {
  let description = [{
    Interface to costumize the how GVN finds equivalence between Operations.
    The default behavior is sufficient for most operations.
    In the GVN framework, hash and comparaison a split into 2 parts, comparing
    the action taken by an Operation and comparing what that actions was applied to.
    This interface allows changing how the GVN framework identifies actions
    as being the same. And also allows to control how the changes are applied
    since the user is can make comparaison where the types donc match.

    Obviously the hash and the compare must be coherent with each other.
    Also the compare (and even the hash) should not match with an operation
    that doesnt implement this interface.
  }];

  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/"return hash of the operation, not its parameters",
      /*retTy=*/"unsigned",
      /*methodName=*/"computeOperationActionHash",
      /*args=*/(ins "::mlir::Operation*": $op, "unsigned": $inputHash)
    >,
    InterfaceMethod<
      /*desc=*/"compare an operation to an other, compare(a, b) == not compare(b, a)",
      /*retTy=*/"bool",
      /*methodName=*/"isOperationActionEqual",
      /*args=*/(ins "::mlir::Operation*": $lhs,
                    "::mlir::Operation*": $rhs,
                    "::llvm::ArrayRef<::llvm::PointerUnion<::mlir::Attribute, ::mlir::Value>>": $lhsOperands,
                    "::llvm::ArrayRef<::llvm::PointerUnion<::mlir::Attribute, ::mlir::Value>>": $rhsOperands)
    >,
    InterfaceMethod<
      /*desc=*/"perform IR edits for an operation with this interface",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"editIRForMatchingValues",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::Value>":$equivalentValues,
                    "::mlir::IRRewriter&": $rewriter)
    >
  ];
}

#endif // MLIR_INTERFACES_GVN_OP_INTERFACE
