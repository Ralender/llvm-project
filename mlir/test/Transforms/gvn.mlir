// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(gvn))' | FileCheck %s

// CHECK-LABEL:   func.func @up_propagate_region() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2(%[[VAL_2]] : i32)
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2(%[[VAL_0]] : i32)
// CHECK:         ^bb2(%[[VAL_3:.*]]: i32):
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @up_propagate_region() -> i32 {
  %1 = arith.constant 0 : i32
  %true = arith.constant true
  cf.cond_br %true, ^bb1, ^bb2(%1 : i32)

  ^bb1:

  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb2(%c1_i32 : i32)

  ^bb2(%arg : i32):

  %c1_i32_0 = arith.constant 1 : i32
  %2 = arith.addi %arg, %c1_i32_0 : i32
  return %2 : i32
}
// CHECK-LABEL:   func.func @basic() -> (i32, i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_2]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_2]], %[[VAL_2]] : i32, i32
// CHECK:         }
func.func @basic() -> (i32, i32) {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.addi %0, %1 : i32
  %3 = arith.addi %1, %0 : i32
  %4 = arith.muli %2, %3 : i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL:   func.func @basic1(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32) -> (i32, i32) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_2]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_0]], %[[VAL_3]] : i32, i32
// CHECK:         }
func.func @basic1(i32) -> (i32, i32) {
^bb(%arg : i32):
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.addi %arg, %0 : i32
  %3 = arith.addi %1, %arg : i32
  %4 = arith.muli %2, %3 : i32
  return %arg, %4 : i32, i32
}

// CHECK-LABEL:   func.func @many(
// CHECK-SAME:                    %[[VAL_0:.*]]: f32,
// CHECK-SAME:                    %[[VAL_1:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_3:.*]] = arith.addf %[[VAL_2]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_4:.*]] = arith.addf %[[VAL_3]], %[[VAL_3]] : f32
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_4]] : f32
// CHECK:           return %[[VAL_5]] : f32
// CHECK:         }
func.func @many(f32, f32) -> (f32) {
^bb0(%a : f32, %b : f32):
  %c = arith.addf %a, %b : f32
  %d = arith.addf %a, %b : f32
  %e = arith.addf %a, %b : f32
  %f = arith.addf %a, %b : f32
  %g = arith.addf %c, %d : f32
  %h = arith.addf %e, %f : f32
  %i = arith.addf %c, %e : f32
  %j = arith.addf %g, %h : f32
  %k = arith.addf %h, %i : f32
  %l = arith.addf %j, %k : f32
  return %l : f32
}

// CHECK-LABEL:   func.func @phi1(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                    %[[VAL_1:.*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i1) -> i32 {
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
func.func @phi1(i32, i1, i1) -> i32 {
^bb(%arg : i32, %c0 : i1, %c1 : i1):
  cf.cond_br %c0, ^bb1(%arg : i32), ^bb2(%arg : i32)

^bb1(%arg1 : i32):
  cf.br ^bb2(%arg1 : i32)

^bb2(%arg2 : i32):
  cf.cond_br %c1, ^bb1(%arg2 : i32), ^bb3(%arg2 : i32)

^bb3(%arg4 : i32):
  func.return %arg4 : i32
}

// CHECK-LABEL:   func.func @phi2(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                    %[[VAL_1:.*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i1) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1(%[[VAL_0]] : i32), ^bb2(%[[VAL_0]] : i32)
// CHECK:         ^bb1(%[[VAL_4:.*]]: i32):
// CHECK:           cf.br ^bb2(%[[VAL_3]] : i32)
// CHECK:         ^bb2(%[[VAL_5:.*]]: i32):
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1(%[[VAL_5]] : i32), ^bb3(%[[VAL_5]] : i32)
// CHECK:         ^bb3(%[[VAL_6:.*]]: i32):
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }
func.func @phi2(i32, i1, i1) -> i32 {
^bb(%arg : i32, %c0 : i1, %c1 : i1):
  cf.cond_br %c0, ^bb1(%arg : i32), ^bb2(%arg : i32)

^bb1(%arg1 : i32):
  %1 = arith.constant 0 : i32
  cf.br ^bb2(%1 : i32)

^bb2(%arg2 : i32):
  cf.cond_br %c1, ^bb1(%arg2 : i32), ^bb3(%arg2 : i32)

^bb3(%arg4 : i32):
  func.return %arg4 : i32
}

// CHECK-LABEL:   func.func @phi3(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                    %[[VAL_1:.*]]: i1,
// CHECK-SAME:                    %[[VAL_2:.*]]: i1) -> i32 {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
func.func @phi3(i32, i1, i1) -> i32 {
^bb(%arg : i32, %c0 : i1, %c1 : i1):
  %0 = arith.constant 0 : i32
  %tmp = arith.addi %0, %arg : i32
  cf.cond_br %c0, ^bb1(%tmp : i32), ^bb2(%tmp : i32)

^bb1(%arg1 : i32):
  cf.br ^bb2(%arg1 : i32)

^bb2(%arg2 : i32):
  cf.cond_br %c1, ^bb1(%arg2 : i32), ^bb3(%arg2 : i32)

^bb3(%arg4 : i32):
  %tmp3 = arith.subi %tmp, %arg4 : i32
  func.return %tmp3 : i32
}

// CHECK-LABEL:   func.func @phi4(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = "foo.i1"() : () -> i1
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2(%[[VAL_0]] : i32)
// CHECK:         ^bb1:
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           cf.br ^bb2(%[[VAL_2]] : i32)
// CHECK:         ^bb2(%[[VAL_3:.*]]: i32):
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_0]] : i32
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @phi4(i32) -> i32 {
^bb(%arg : i32):
  %c0 = "foo.i1"() {} : () -> i1
  cf.cond_br %c0, ^bb1(%arg : i32), ^bb2(%arg : i32)

^bb1(%arg1 : i32):
  %1 = arith.addi %arg1 , %arg : i32
  cf.br ^bb2(%1 : i32)

^bb2(%arg2 : i32):
  %2 = arith.addi %arg2 , %arg : i32
  cf.br ^bb3(%2 : i32)

^bb3(%arg4 : i32):
  func.return %arg4 : i32
}

// CHECK-LABEL:   func.func @phi5(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_2:.*]] = "foo.i1"() : () -> i1
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_3:.*]] = "foo.test"(%[[VAL_1]]) : (i32) -> i1
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[VAL_1]] : i32
// CHECK:         }
func.func @phi5(i32) -> i32 {
^bb(%arg : i32):
  %2 = arith.addi %arg , %arg : i32
  %c0 = "foo.i1"() : () -> i1
  cf.cond_br %c0, ^bb1(%2 : i32), ^bb2(%2 : i32)

^bb1(%arg1 : i32):
  cf.br ^bb2(%arg1 : i32)

^bb2(%arg2 : i32):
  "foo.test"(%arg2) : (i32) -> i1
  cf.br ^bb3(%arg2 : i32)

^bb3(%arg4 : i32):
  %arg5 = arith.addi %arg , %arg : i32
  func.return %arg5 : i32
}

// CHECK-LABEL:   func.func @phi6(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.addi %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_2:.*]] = "foo.i1"() : () -> i1
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2(%[[VAL_1]] : i32)
// CHECK:         ^bb1:
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_1]] : i32
// CHECK:           cf.br ^bb2(%[[VAL_3]] : i32)
// CHECK:         ^bb2(%[[VAL_4:.*]]: i32):
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[VAL_4]] : i32
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : i32
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }
func.func @phi6(i32) -> i32 {
^bb(%arg : i32):
  %0 = arith.addi %arg , %arg : i32
  %c0 = "foo.i1"() : () -> i1
  cf.cond_br %c0, ^bb1(%0 : i32), ^bb2(%0 : i32)

^bb1(%arg1 : i32):
  %1 = arith.addi %arg1 , %arg1 : i32
  cf.br ^bb2(%1 : i32)

^bb2(%arg2 : i32):
  %2 = arith.addi %arg2 , %arg2 : i32
  cf.br ^bb3(%2 : i32)

^bb3(%arg4 : i32):
  %arg5 = arith.addi %arg4 , %arg4 : i32
  func.return %arg5 : i32
}

// CHECK-LABEL:   func.func @unreachable_use(
// CHECK-SAME:                               %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = "foo.op"() : () -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32):
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_3]] : i32
// CHECK:         }
func.func @unreachable_use(i32) -> i32 {
^bb(%arg : i32):
  %c = "foo.op"() : () -> i32
  func.return %arg : i32

^bb1(%c1 : i32):
  %res = arith.addi %c1, %c : i32
  func.return %res : i32
}

// CHECK-LABEL:   func.func @phi7(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = "foo.i1"() : () -> i1
// CHECK:           %[[VAL_2:.*]] = "foo.op"() : () -> i32
// CHECK:           %[[VAL_3:.*]] = "foo.op"() : () -> i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[VAL_3]] : i32)
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3(%[[VAL_2]] : i32)
// CHECK:         ^bb3(%[[VAL_4:.*]]: i32):
// CHECK:           %[[VAL_5:.*]] = "foo.i1"() : () -> i1
// CHECK:           cf.cond_br %[[VAL_5]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           cf.br ^bb6(%[[VAL_3]] : i32)
// CHECK:         ^bb5:
// CHECK:           cf.br ^bb6(%[[VAL_2]] : i32)
// CHECK:         ^bb6(%[[VAL_6:.*]]: i32):
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_4]] : i32
// CHECK:           return %[[VAL_7]] : i32
// CHECK:         }
func.func @phi7(i32) -> i32 {
^bb(%arg : i32):
  %cond0 = "foo.i1"() : () -> i1
  %b = "foo.op"() : () -> i32
  %a = "foo.op"() : () -> i32
  cf.cond_br %cond0, ^bb1, ^bb2

^bb1():
  cf.br ^bb3(%a : i32)

^bb2():
  cf.br ^bb3(%b : i32)

^bb3(%c : i32):
  %cond1 = "foo.i1"() : () -> i1
  cf.cond_br %cond1, ^bb4, ^bb5

^bb4():
  cf.br ^bb6(%a : i32)

^bb5():
  cf.br ^bb6(%b : i32)

^bb6(%c1 : i32):
  %res = arith.addi %c1, %c : i32
  func.return %res : i32
}

// CHECK-LABEL:   func.func @phi8(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = "foo.i1"() : () -> i1
// CHECK:           %[[VAL_2:.*]] = "foo.op"() : () -> i32
// CHECK:           %[[VAL_3:.*]] = "foo.op"() : () -> i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3(%[[VAL_3]], %[[VAL_2]] : i32, i32)
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3(%[[VAL_2]], %[[VAL_3]] : i32, i32)
// CHECK:         ^bb3(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:           %[[VAL_6:.*]] = arith.addi %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           return %[[VAL_6]] : i32
// CHECK:         }
func.func @phi8(i32) -> i32 {
^bb(%arg : i32):
  %cond0 = "foo.i1"() : () -> i1
  %b = "foo.op"() : () -> i32
  %a = "foo.op"() : () -> i32
  cf.cond_br %cond0, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%a, %b : i32, i32)

^bb2:
  cf.br ^bb3(%b, %a : i32, i32)

^bb3(%ab : i32, %ba : i32):
  %res = arith.addi %ab, %ba : i32
  func.return %res : i32
}

// CHECK-LABEL:   func.func @phi9(
// CHECK-SAME:                    %[[VAL_0:.*]]: i32) -> (i32, i32) {
// CHECK:           %[[VAL_1:.*]] = "foo.i1"() : () -> i1
// CHECK:           %[[VAL_2:.*]] = "foo.op"() : () -> i32
// CHECK:           %[[VAL_3:.*]] = "foo.op"() : () -> i32
// CHECK:           cf.cond_br %[[VAL_1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_4]], %[[VAL_5]] : i32, i32
// CHECK:         }
func.func @phi9(i32) -> (i32, i32) {
^bb(%arg : i32):
  %cond0 = "foo.i1"() : () -> i1
  %b = "foo.op"() : () -> i32
  %a = "foo.op"() : () -> i32
  cf.cond_br %cond0, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%b, %a : i32, i32)

^bb2:
  cf.br ^bb3(%b, %a : i32, i32)

^bb3(%ab : i32, %ba : i32):
  %res1 = arith.addi %ab, %ab : i32
  %res2 = arith.addi %ba, %ab : i32
  func.return %res1, %res2 : i32, i32
}

/// The edits GVN did are not really important but this used to crash
func.func @crash0(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
  %1 = arith.cmpi slt, %0, %c3 : index
  cf.cond_br %1, ^bb2, ^bb6
^bb2:  // pred: ^bb1
  %c0_0 = arith.constant 0 : index
  %c3_1 = arith.constant 3 : index
  %c1_2 = arith.constant 1 : index
  cf.br ^bb3(%c0_0 : index)
^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
  %3 = arith.cmpi slt, %2, %c3_1 : index
  cf.cond_br %3, ^bb4, ^bb5
^bb4:  // pred: ^bb3
  memref.store %cst, %alloc[%0, %2] : memref<3x3xf32>
  %4 = arith.addi %2, %c1_2 : index
  cf.br ^bb3(%4 : index)
^bb5:  // pred: ^bb3
  %5 = arith.addi %0, %c1 : index
  cf.br ^bb1(%5 : index)
^bb6:  // pred: ^bb1
  %c0_3 = arith.constant 0 : index
  %c3_4 = arith.constant 3 : index
  %c1_5 = arith.constant 1 : index
  cf.br ^bb7(%c0_3 : index)
^bb7(%6: index):  // 2 preds: ^bb6, ^bb14
  %7 = arith.cmpi slt, %6, %c3_4 : index
  cf.cond_br %7, ^bb8, ^bb15
^bb8:  // pred: ^bb7
  %c0_6 = arith.constant 0 : index
  %c3_7 = arith.constant 3 : index
  %c1_8 = arith.constant 1 : index
  cf.br ^bb9(%c0_6 : index)
^bb9(%8: index):  // 2 preds: ^bb8, ^bb13
  %9 = arith.cmpi slt, %8, %c3_7 : index
  cf.cond_br %9, ^bb10, ^bb14
^bb10:  // pred: ^bb9
  %c0_9 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1_10 = arith.constant 1 : index
  cf.br ^bb11(%c0_9 : index)
^bb11(%10: index):  // 2 preds: ^bb10, ^bb12
  %11 = arith.cmpi slt, %10, %c4 : index
  cf.cond_br %11, ^bb12, ^bb13
^bb12:  // pred: ^bb11
  %12 = memref.load %arg1[%10, %8] : memref<4x3xf32>
  %13 = memref.load %arg0[%6, %10] : memref<3x4xf32>
  %14 = arith.mulf %13, %12 : f32
  %15 = memref.load %alloc[%6, %8] : memref<3x3xf32>
  %16 = arith.addf %15, %14 : f32
  memref.store %16, %alloc[%6, %8] : memref<3x3xf32>
  %17 = arith.addi %10, %c1_10 : index
  cf.br ^bb11(%17 : index)
^bb13:  // pred: ^bb11
  %18 = arith.addi %8, %c1_8 : index
  cf.br ^bb9(%18 : index)
^bb14:  // pred: ^bb9
  %19 = arith.addi %6, %c1_5 : index
  cf.br ^bb7(%19 : index)
^bb15:  // pred: ^bb7
  %c0_11 = arith.constant 0 : index
  %c3_12 = arith.constant 3 : index
  %c1_13 = arith.constant 1 : index
  cf.br ^bb16(%c0_11 : index)
^bb16(%20: index):  // 2 preds: ^bb15, ^bb20
  %21 = arith.cmpi slt, %20, %c3_12 : index
  cf.cond_br %21, ^bb17, ^bb21
^bb17:  // pred: ^bb16
  %c0_14 = arith.constant 0 : index
  %c3_15 = arith.constant 3 : index
  %c1_16 = arith.constant 1 : index
  cf.br ^bb18(%c0_14 : index)
^bb18(%22: index):  // 2 preds: ^bb17, ^bb19
  %23 = arith.cmpi slt, %22, %c3_15 : index
  cf.cond_br %23, ^bb19, ^bb20
^bb19:  // pred: ^bb18
  %24 = memref.load %arg2[%20, %22] : memref<3x3xf32>
  %25 = memref.load %alloc[%20, %22] : memref<3x3xf32>
  %26 = arith.addf %25, %24 : f32
  memref.store %26, %arg3[%20, %22] : memref<3x3xf32>
  %27 = arith.addi %22, %c1_16 : index
  cf.br ^bb18(%27 : index)
^bb20:  // pred: ^bb18
  %28 = arith.addi %20, %c1_13 : index
  cf.br ^bb16(%28 : index)
^bb21:  // pred: ^bb16
  return
}
