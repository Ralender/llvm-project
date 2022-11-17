//===- GvnOpInterface.h - interface to customize GVN behavior for an Op ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Most ops do not need special treatment so they do not need to customize The
// behavior of GVN.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_GVN_OP_INTERFACE_H_
#define MLIR_INTERFACES_GVN_OP_INTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

/// Include the generated interface declarations.
#include "mlir/Interfaces/GvnOpInterface.h.inc"

#endif // MLIR_INTERFACES_GVN_OP_INTERFACE_H_
