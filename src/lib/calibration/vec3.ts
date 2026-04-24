/**
 * 3D vector operations.
 * All vectors represented as {x, y, z} objects for clarity.
 */

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface Vec2 {
  x: number
  y: number
}

/** 6-element vector for B matrix coefficients [B11, B12, B22, B13, B23, B33] */
export type Vec6 = [number, number, number, number, number, number]

export const Vec3 = {
  /** Create vector from components */
  of(x: number, y: number, z: number): Vec3 {
    return { x, y, z }
  },

  /** Create zero vector */
  zero(): Vec3 {
    return { x: 0, y: 0, z: 0 }
  },

  /** Create unit vector along axis */
  unit(axis: 'x' | 'y' | 'z'): Vec3 {
    switch (axis) {
      case 'x':
        return { x: 1, y: 0, z: 0 }
      case 'y':
        return { x: 0, y: 1, z: 0 }
      case 'z':
        return { x: 0, y: 0, z: 1 }
    }
  },

  /** Clone a vector */
  clone(v: Vec3): Vec3 {
    return { x: v.x, y: v.y, z: v.z }
  },

  /** Add two vectors: a + b */
  add(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
  },

  /** Subtract vectors: a - b */
  sub(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }
  },

  /** Scale vector by scalar */
  scale(v: Vec3, s: number): Vec3 {
    return { x: v.x * s, y: v.y * s, z: v.z * s }
  },

  /** Negate vector */
  negate(v: Vec3): Vec3 {
    return { x: -v.x, y: -v.y, z: -v.z }
  },

  /** Dot product: a · b */
  dot(a: Vec3, b: Vec3): number {
    return a.x * b.x + a.y * b.y + a.z * b.z
  },

  /** Squared length of vector */
  lengthSq(v: Vec3): number {
    return v.x * v.x + v.y * v.y + v.z * v.z
  },

  /** Length (L2 norm) of vector */
  length(v: Vec3): number {
    return Math.sqrt(Vec3.lengthSq(v))
  },

  /** Normalize vector to unit length */
  normalize(v: Vec3): Vec3 {
    const len = Vec3.length(v)
    if (len === 0) return { x: 0, y: 0, z: 0 }
    return Vec3.scale(v, 1 / len)
  },

  /** Cross product: a × b */
  cross(a: Vec3, b: Vec3): Vec3 {
    return {
      x: a.y * b.z - a.z * b.y,
      y: a.z * b.x - a.x * b.z,
      z: a.x * b.y - a.y * b.x,
    }
  },

  /** Element-wise multiply */
  mul(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x * b.x, y: a.y * b.y, z: a.z * b.z }
  },

  /** Element-wise divide */
  div(a: Vec3, b: Vec3): Vec3 {
    return { x: a.x / b.x, y: a.y / b.y, z: a.z / b.z }
  },

  /** Linear interpolation: a + t*(b - a) */
  lerp(a: Vec3, b: Vec3, t: number): Vec3 {
    return Vec3.add(a, Vec3.scale(Vec3.sub(b, a), t))
  },

  /** Distance between two vectors */
  distance(a: Vec3, b: Vec3): number {
    return Vec3.length(Vec3.sub(a, b))
  },

  /** Squared distance between two vectors */
  distanceSq(a: Vec3, b: Vec3): number {
    return Vec3.lengthSq(Vec3.sub(a, b))
  },

  /** Check if two vectors are equal (within tolerance) */
  equals(a: Vec3, b: Vec3, tol = 1e-12): boolean {
    return (
      Math.abs(a.x - b.x) <= tol &&
      Math.abs(a.y - b.y) <= tol &&
      Math.abs(a.z - b.z) <= tol
    )
  },

  /** Convert to flat array [x, y, z] */
  toArray(v: Vec3): [number, number, number] {
    return [v.x, v.y, v.z]
  },

  /** Create from flat array [x, y, z] */
  fromArray(arr: readonly number[]): Vec3 {
    return { x: arr[0]!, y: arr[1]!, z: arr[2]! }
  },

  /** Apply function to each component */
  map(v: Vec3, f: (c: number) => number): Vec3 {
    return { x: f(v.x), y: f(v.y), z: f(v.z) }
  },

  /** Sum of components */
  sum(v: Vec3): number {
    return v.x + v.y + v.z
  },

  /** Absolute value of each component */
  abs(v: Vec3): Vec3 {
    return { x: Math.abs(v.x), y: Math.abs(v.y), z: Math.abs(v.z) }
  },
}

// Extend builtin Math for Vec3 methods
declare global {
  interface Math {
    /** Vec3 dot product */
    vec3dot(a: Vec3, b: Vec3): number
  }
}

// Math-based operations (static, not modifying prototype)
export const vec3dot = Vec3.dot
export const vec3cross = Vec3.cross
export const vec3norm = Vec3.length
export const vec3normalize = Vec3.normalize
