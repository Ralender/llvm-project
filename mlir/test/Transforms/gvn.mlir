// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(gvn))' | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(gvn{hash-collide=1}))' | FileCheck %s

// CHECK-LABEL:   func.func @basic_constant_fold() -> (i32, i32) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           return %[[VAL_2]], %[[VAL_2]] : i32, i32
// CHECK:         }
func.func @basic_constant_fold() -> (i32, i32) {
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

/// TODO: cleanup dead constants
// CHECK-LABEL:   func.func @cf_constant_fold() -> i32 {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
func.func @cf_constant_fold() -> i32 {
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

/// Sadly mul(a, 0) is not folded to 0 via the fold API
// CHECK-LABEL:   func.func @control_flow_fold(
// CHECK-SAME:                                 %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant false
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           cf.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @control_flow_fold(%arg : i32) -> i32 {
  %false = arith.constant false
  %c1_i32_0 = arith.constant 0 : i32
  cf.cond_br %false, ^bb1, ^bb2(%arg, %c1_i32_0 : i32, i32)

^bb1:
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb2(%c1_i32, %c1_i32 : i32, i32)

^bb2(%arg1 : i32, %arg2: i32):
  %2 = arith.muli %arg1, %arg2 : i32
  return %2 : i32
}

// CHECK-LABEL:   func.func @control_flow_fold2(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i32) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           cf.cond_br %[[VAL_1]], ^bb2, ^bb1
// CHECK:         ^bb1:
// CHECK:           cf.cond_br %[[VAL_1]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_2:.*]] = "foo.i1"() : () -> i1
// CHECK:           %[[VAL_3:.*]] = "foo.i32"() : () -> i32
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1, ^bb2
// CHECK:         ^bb3:
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_0]], %[[VAL_0]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
func.func @control_flow_fold2(%arg : i32) -> i32 {
  %false = arith.constant false
  cf.cond_br %false, ^bb3, ^bb1(%false, %arg : i1, i32)

^bb1(%c : i1, %a : i32):
  cf.cond_br %c, ^bb3, ^bb2(%a : i32)

^bb3:
  %c0 = "foo.i1"() : () -> i1
  %arg2 = "foo.i32"() : () -> i32
  cf.cond_br %c0, ^bb1(%c0, %arg2 : i1, i32), ^bb3

^bb2(%arg1 : i32):
  %2 = arith.muli %arg1, %arg : i32
  return %2 : i32
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
// CHECK:           cf.cond_br %[[VAL_2]], ^bb1(%[[VAL_5]] : i32), ^bb3
// CHECK:         ^bb3:
// CHECK:           return %[[VAL_5]] : i32
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

/// Check that operations are not eliminated if they have different result
/// types.
// CHECK-LABEL: @different_results
func.func @different_results(%arg0: tensor<*xf32>) -> (tensor<?x?xf32>, tensor<4x?xf32>) {
  // CHECK: %[[VAR_0:[0-9a-zA-Z_]+]] = tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?xf32>
  // CHECK-NEXT: %[[VAR_1:[0-9a-zA-Z_]+]] = tensor.cast %{{.*}} : tensor<*xf32> to tensor<4x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %1 = tensor.cast %arg0 : tensor<*xf32> to tensor<4x?xf32>

  // CHECK-NEXT: return %[[VAR_0]], %[[VAR_1]] : tensor<?x?xf32>, tensor<4x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<4x?xf32>
}

/// The edits GVN did are not really important but this used to crash
// CHECK-LABEL:   func.func @crash0(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<3x4xf32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<4x3xf32>,
// CHECK-SAME:                      %[[VAL_2:.*]]: memref<3x3xf32>,
// CHECK-SAME:                      %[[VAL_3:.*]]: memref<3x3xf32>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           cf.br ^bb1(%[[VAL_7]] : index)
// CHECK:         ^bb1(%[[VAL_10:.*]]: index):
// CHECK:           %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_11]], ^bb2, ^bb6
// CHECK:         ^bb2:
// CHECK:           cf.br ^bb3(%[[VAL_7]] : index)
// CHECK:         ^bb3(%[[VAL_12:.*]]: index):
// CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_13]], ^bb4, ^bb5
// CHECK:         ^bb4:
// CHECK:           memref.store %[[VAL_8]], %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb3(%[[VAL_14]] : index)
// CHECK:         ^bb5:
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_10]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb1(%[[VAL_15]] : index)
// CHECK:         ^bb6:
// CHECK:           cf.br ^bb7(%[[VAL_7]] : index)
// CHECK:         ^bb7(%[[VAL_16:.*]]: index):
// CHECK:           %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_17]], ^bb8, ^bb15
// CHECK:         ^bb8:
// CHECK:           cf.br ^bb9(%[[VAL_7]] : index)
// CHECK:         ^bb9(%[[VAL_18:.*]]: index):
// CHECK:           %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_19]], ^bb10, ^bb14
// CHECK:         ^bb10:
// CHECK:           cf.br ^bb11(%[[VAL_7]] : index)
// CHECK:         ^bb11(%[[VAL_20:.*]]: index):
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_4]] : index
// CHECK:           cf.cond_br %[[VAL_21]], ^bb12, ^bb13
// CHECK:         ^bb12:
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_20]], %[[VAL_18]]] : memref<4x3xf32>
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_16]], %[[VAL_20]]] : memref<3x4xf32>
// CHECK:           %[[VAL_24:.*]] = arith.mulf %[[VAL_23]], %[[VAL_22]] : f32
// CHECK:           %[[VAL_25:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]], %[[VAL_18]]] : memref<3x3xf32>
// CHECK:           %[[VAL_26:.*]] = arith.addf %[[VAL_25]], %[[VAL_24]] : f32
// CHECK:           memref.store %[[VAL_26]], %[[VAL_9]]{{\[}}%[[VAL_16]], %[[VAL_18]]] : memref<3x3xf32>
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_20]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb11(%[[VAL_27]] : index)
// CHECK:         ^bb13:
// CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_18]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb9(%[[VAL_28]] : index)
// CHECK:         ^bb14:
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_16]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb7(%[[VAL_29]] : index)
// CHECK:         ^bb15:
// CHECK:           cf.br ^bb16(%[[VAL_7]] : index)
// CHECK:         ^bb16(%[[VAL_30:.*]]: index):
// CHECK:           %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_31]], ^bb17, ^bb21
// CHECK:         ^bb17:
// CHECK:           cf.br ^bb18(%[[VAL_7]] : index)
// CHECK:         ^bb18(%[[VAL_32:.*]]: index):
// CHECK:           %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_6]] : index
// CHECK:           cf.cond_br %[[VAL_33]], ^bb19, ^bb20
// CHECK:         ^bb19:
// CHECK:           %[[VAL_34:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_30]], %[[VAL_32]]] : memref<3x3xf32>
// CHECK:           %[[VAL_35:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_30]], %[[VAL_32]]] : memref<3x3xf32>
// CHECK:           %[[VAL_36:.*]] = arith.addf %[[VAL_35]], %[[VAL_34]] : f32
// CHECK:           memref.store %[[VAL_36]], %[[VAL_3]]{{\[}}%[[VAL_30]], %[[VAL_32]]] : memref<3x3xf32>
// CHECK:           %[[VAL_37:.*]] = arith.addi %[[VAL_32]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb18(%[[VAL_37]] : index)
// CHECK:         ^bb20:
// CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_30]], %[[VAL_5]] : index
// CHECK:           cf.br ^bb16(%[[VAL_38]] : index)
// CHECK:         ^bb21:
// CHECK:           return
// CHECK:         }
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

// CHECK-LABEL:   func.func @phi10(
// CHECK-SAME:                     %[[VAL_0:.*]]: i32,
// CHECK-SAME:                     %[[VAL_1:.*]]: i32) -> i32 {
// CHECK:           cf.br ^bb1(%[[VAL_0]], %[[VAL_1]] : i32, i32)
// CHECK:         ^bb1(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:           %[[VAL_4:.*]] = "foo.i1"() : () -> i1
// CHECK:           cf.cond_br %[[VAL_4]], ^bb1(%[[VAL_3]], %[[VAL_2]] : i32, i32), ^bb2
// CHECK:         ^bb2:
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }
func.func @phi10(%arg0 : i32, %arg1 : i32) -> i32 {
cf.br ^bb1(%arg0, %arg1 : i32, i32)

^bb1(%a : i32, %b : i32):
  %c0 = "foo.i1"() : () -> i1
  cf.cond_br %c0, ^bb1(%b, %a : i32, i32), ^bb3(%a : i32)

^bb3(%ret : i32):
  func.return %ret: i32
}

/// This does not IR edits, just check that regions are handled properly
// CHECK-LABEL:   func.func @check_regions() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<10xf32>
// CHECK:           affine.for %[[VAL_2:.*]] = 0 to 10 {
// CHECK:             affine.store %[[VAL_0]], %[[VAL_1]]{{\[}}%[[VAL_2]]] : memref<10xf32>
// CHECK:           }
// CHECK:           affine.for %[[VAL_3:.*]] = 0 to 5 {
// CHECK:             affine.for %[[VAL_4:.*]] = 0 to 10 {
// CHECK:               %[[VAL_5:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_4]]] : memref<10xf32>
// CHECK:               affine.store %[[VAL_5]], %[[VAL_1]]{{\[}}%[[VAL_4]]] : memref<10xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @check_regions() {
  %a = memref.alloc() : memref<10xf32>

  %cf0 = arith.constant 0.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cf0, %a[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 5 {
    affine.for %i2 = 0 to 10 {
      %v0 = affine.load %a[%i2] : memref<10xf32>
      affine.store %v0, %a[%i2] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL:   llvm.func @fold_inplace() {
// CHECK:           %[[VAL_0:.*]] = "foo.op"() : () -> !llvm.ptr<i8>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = "foo.op"() : () -> !llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_2]], 0] : (!llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = "foo.op"() : () -> !llvm.ptr<func<void (ptr<struct<"class.std::ios_base::Init", (i8)>>)>>
// CHECK:           %[[VAL_6:.*]] = llvm.bitcast %[[VAL_5]] : !llvm.ptr<func<void (ptr<struct<"class.std::ios_base::Init", (i8)>>)>> to !llvm.ptr<func<void (ptr<i8>)>>
// CHECK:           %[[VAL_7:.*]] = "foo.op"() : () -> !llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>
// CHECK:           "foo.op"(%[[VAL_7]]) : (!llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>) -> ()
// CHECK:           %[[VAL_8:.*]] = "foo.op"(%[[VAL_6]], %[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
// CHECK:           llvm.return
// CHECK:         }
llvm.func @fold_inplace() {
  %0 = "foo.op"() : () -> !llvm.ptr<i8>
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.constant(0 : i32) : i32
  %3 = "foo.op"() : () -> !llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>
  %4 = llvm.getelementptr %3[%2, 0] : (!llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>, i32) -> !llvm.ptr<i8>
  %5 = "foo.op"() : () ->  !llvm.ptr<func<void (ptr<struct<"class.std::ios_base::Init", (i8)>>)>>
  %6 = llvm.bitcast %5 : !llvm.ptr<func<void (ptr<struct<"class.std::ios_base::Init", (i8)>>)>> to !llvm.ptr<func<void (ptr<i8>)>>
  %7 = "foo.op"() : () ->  !llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>
  "foo.op"(%7) : (!llvm.ptr<struct<"class.std::ios_base::Init", (i8)>>) -> ()
  %8 = "foo.op"(%6, %4, %0) : (!llvm.ptr<func<void (ptr<i8>)>>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
  llvm.return
}
