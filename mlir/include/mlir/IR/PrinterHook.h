//===- PrinterHook.h - Printer hook for annotations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the base class of Annotation printer hooks
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_PRINTER_HOOK_H
#define MLIR_IR_PRINTER_HOOK_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class Block;

/// Printer Hook to add annotations before an after IR constructs.
/// It allows analysis to print there results directly within the IR making it
/// easier for human to read analysis results.
class PrinterHookBase {
public:
  /// Every annotations should be a self-contained comment. so it must start
  /// with "//" (maybe after indenting) and finish with a newline. otherwise it
  /// will likely not be possible to parse the generated IR.
  virtual void printCommentBeforeOp(Operation *op, raw_ostream &os,
                                    unsigned currentIndent) = 0;
  virtual void printCommentBeforeBlock(Block *block, raw_ostream &os,
                                       unsigned currentIndent) = 0;
  virtual ~PrinterHookBase();
};

} // namespace mlir

#endif
