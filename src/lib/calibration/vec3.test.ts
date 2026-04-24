import { describe, expect, test } from 'vitest'
import { Vec3, vec3dot, vec3cross, vec3norm, vec3normalize } from './vec3'

describe('Vec3', () => {
  describe('construction', () => {
    test('of creates vector from components', () => {
      expect(Vec3.of(1, 2, 3)).toEqual({ x: 1, y: 2, z: 3 })
    })

    test('zero creates zero vector', () => {
      expect(Vec3.zero()).toEqual({ x: 0, y: 0, z: 0 })
    })

    test('unit creates unit vectors', () => {
      expect(Vec3.unit('x')).toEqual({ x: 1, y: 0, z: 0 })
      expect(Vec3.unit('y')).toEqual({ x: 0, y: 1, z: 0 })
      expect(Vec3.unit('z')).toEqual({ x: 0, y: 0, z: 1 })
    })

    test('clone creates copy', () => {
      const v = Vec3.of(1, 2, 3)
      const c = Vec3.clone(v)
      expect(c).toEqual(v)
      expect(c).not.toBe(v) // different object
    })
  })

  describe('arithmetic', () => {
    test('add combines vectors', () => {
      expect(Vec3.add(Vec3.of(1, 2, 3), Vec3.of(4, 5, 6))).toEqual(Vec3.of(5, 7, 9))
    })

    test('sub subtracts vectors', () => {
      expect(Vec3.sub(Vec3.of(4, 5, 6), Vec3.of(1, 2, 3))).toEqual(Vec3.of(3, 3, 3))
    })

    test('scale multiplies by scalar', () => {
      expect(Vec3.scale(Vec3.of(1, 2, 3), 2.5)).toEqual(Vec3.of(2.5, 5, 7.5))
    })

    test('negate flips sign', () => {
      expect(Vec3.negate(Vec3.of(1, -2, 3))).toEqual(Vec3.of(-1, 2, -3))
    })

    test('mul does element-wise multiply', () => {
      expect(Vec3.mul(Vec3.of(2, 3, 4), Vec3.of(5, 6, 7))).toEqual(Vec3.of(10, 18, 28))
    })

    test('div does element-wise divide', () => {
      expect(Vec3.div(Vec3.of(10, 18, 28), Vec3.of(2, 3, 4))).toEqual(Vec3.of(5, 6, 7))
    })
  })

  describe('dot product', () => {
    test('dot product of orthogonal vectors is 0', () => {
      expect(Vec3.dot(Vec3.unit('x'), Vec3.unit('y'))).toBe(0)
      expect(Vec3.dot(Vec3.unit('x'), Vec3.unit('z'))).toBe(0)
      expect(Vec3.dot(Vec3.unit('y'), Vec3.unit('z'))).toBe(0)
    })

    test('dot product of parallel vectors equals product of lengths', () => {
      const v = Vec3.of(1, 2, 3)
      expect(Vec3.dot(v, v)).toBeCloseTo(Vec3.lengthSq(v))
    })

    test('dot product with self equals squared length', () => {
      expect(Vec3.dot(Vec3.of(3, 4, 0), Vec3.of(3, 4, 0))).toBe(25)
    })

    test('dot product is commutative', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.dot(a, b)).toBeCloseTo(Vec3.dot(b, a))
    })

    test('dot product is distributive', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.of(7, 8, 9)
      const ab = Vec3.dot(a, b)
      const ac = Vec3.dot(a, c)
      const a_bc = Vec3.dot(a, Vec3.add(b, c))
      expect(a_bc).toBeCloseTo(ab + ac)
    })

    test('dot product is linear in first argument', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const s = 2.5
      expect(Vec3.dot(Vec3.scale(a, s), b)).toBeCloseTo(s * Vec3.dot(a, b))
    })

    test('static vec3dot function works', () => {
      expect(vec3dot(Vec3.of(1, 2, 3), Vec3.of(4, 5, 6))).toBe(32)
    })
  })

  describe('cross product', () => {
    test('cross product of parallel vectors is zero', () => {
      const v = Vec3.of(1, 2, 3)
      expect(Vec3.cross(v, v)).toEqual(Vec3.zero())
      expect(Vec3.cross(v, Vec3.scale(v, 2))).toEqual(Vec3.zero())
    })

    test('cross product is anti-commutative', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.cross(a, b)).toEqual(Vec3.negate(Vec3.cross(b, a)))
    })

    test('cross product of unit vectors follows right-hand rule', () => {
      expect(Vec3.cross(Vec3.unit('x'), Vec3.unit('y'))).toEqual(Vec3.unit('z'))
      expect(Vec3.cross(Vec3.unit('y'), Vec3.unit('z'))).toEqual(Vec3.unit('x'))
      expect(Vec3.cross(Vec3.unit('z'), Vec3.unit('x'))).toEqual(Vec3.unit('y'))
    })

    test('cross product is perpendicular to both inputs', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.cross(a, b)
      expect(Vec3.dot(a, c)).toBeCloseTo(0)
      expect(Vec3.dot(b, c)).toBeCloseTo(0)
    })

    test('cross product magnitude equals |a||b|sin(theta)', () => {
      const a = Vec3.of(1, 0, 0)
      const b = Vec3.of(0, 1, 0)
      expect(Vec3.length(Vec3.cross(a, b))).toBeCloseTo(1)

      // 45 degree angle
      // c = (1,1,0), |c| = sqrt(2)
      // d = (1,0,0), |d| = 1
      // |c × d| = |c||d|sin(45) = sqrt(2) * sin(45) = sqrt(2) * sqrt(2)/2 = 1
      const c = Vec3.of(1, 1, 0)
      const d = Vec3.of(1, 0, 0)
      expect(Vec3.length(Vec3.cross(c, d))).toBeCloseTo(1)
    })

    test('cross product satisfies BAC-CAB identity', () => {
      // BAC-CAB: a × (b × c) = b(a·c) - c(a·b)
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.of(7, 8, 9)
      const lhs = Vec3.cross(a, Vec3.cross(b, c))
      const rhs = Vec3.sub(Vec3.scale(b, Vec3.dot(a, c)), Vec3.scale(c, Vec3.dot(a, b)))
      expect(Vec3.length(Vec3.sub(lhs, rhs))).toBeLessThan(1e-10)
    })

    test('static vec3cross function works', () => {
      expect(vec3cross(Vec3.of(1, 2, 3), Vec3.of(4, 5, 6))).toEqual(Vec3.of(-3, 6, -3))
    })
  })

  describe('length and norm', () => {
    test('length of zero vector is 0', () => {
      expect(Vec3.length(Vec3.zero())).toBe(0)
    })

    test('length of unit vectors is 1', () => {
      expect(Vec3.length(Vec3.unit('x'))).toBe(1)
      expect(Vec3.length(Vec3.unit('y'))).toBe(1)
      expect(Vec3.length(Vec3.unit('z'))).toBe(1)
    })

    test('length of (3,4,0) is 5 (3-4-5 triangle)', () => {
      expect(Vec3.length(Vec3.of(3, 4, 0))).toBe(5)
    })

    test('length is always non-negative', () => {
      for (const v of [
        Vec3.of(1, 2, 3),
        Vec3.of(-1, 2, -3),
        Vec3.of(0, -5, 0),
        Vec3.unit('x'),
      ]) {
        expect(Vec3.length(v)).toBeGreaterThanOrEqual(0)
      }
    })

    test('lengthSq equals length squared', () => {
      const v = Vec3.of(3, 4, 5)
      expect(Vec3.lengthSq(v)).toBeCloseTo(Vec3.length(v) ** 2)
    })

    test('static vec3norm function works', () => {
      expect(vec3norm(Vec3.of(3, 4, 0))).toBe(5)
    })
  })

  describe('normalization', () => {
    test('normalize of zero vector is zero vector', () => {
      expect(Vec3.normalize(Vec3.zero())).toEqual(Vec3.zero())
    })

    test('normalize makes length 1', () => {
      const v = Vec3.of(3, 4, 0)
      expect(Vec3.length(Vec3.normalize(v))).toBeCloseTo(1)
    })

    test('normalize preserves direction', () => {
      const v = Vec3.of(3, 4, 0)
      const n = Vec3.normalize(v)
      // Both should point in same direction (ratio of components preserved)
      expect(n.x / 3).toBeCloseTo(n.y / 4)
      // z component should remain 0
      expect(n.z).toBeCloseTo(0)
    })

    test('normalize of unit vector is unchanged', () => {
      const u = Vec3.unit('x')
      expect(Vec3.normalize(u)).toEqual(u)
    })

    test('static vec3normalize function works', () => {
      const result = vec3normalize(Vec3.of(3, 4, 0))
      expect(result.x).toBeCloseTo(0.6)
      expect(result.y).toBeCloseTo(0.8)
      expect(result.z).toBeCloseTo(0)
    })
  })

  describe('lerp', () => {
    test('lerp at t=0 returns a', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.lerp(a, b, 0)).toEqual(a)
    })

    test('lerp at t=1 returns b', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.lerp(a, b, 1)).toEqual(b)
    })

    test('lerp at t=0.5 returns midpoint', () => {
      const a = Vec3.of(0, 0, 0)
      const b = Vec3.of(10, 20, 30)
      expect(Vec3.lerp(a, b, 0.5)).toEqual(Vec3.of(5, 10, 15))
    })

    test('lerp is linear', () => {
      const a = Vec3.of(0, 0, 0)
      const b = Vec3.of(10, 20, 30)
      expect(Vec3.lerp(a, b, 0.25)).toEqual(Vec3.of(2.5, 5, 7.5))
      expect(Vec3.lerp(a, b, 0.75)).toEqual(Vec3.of(7.5, 15, 22.5))
    })
  })

  describe('distance', () => {
    test('distance from a to a is 0', () => {
      const a = Vec3.of(1, 2, 3)
      expect(Vec3.distance(a, a)).toBe(0)
    })

    test('distance is symmetric', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 6, 9)
      expect(Vec3.distance(a, b)).toBeCloseTo(Vec3.distance(b, a))
    })

    test('distance matches Euclidean distance', () => {
      const a = Vec3.of(0, 0, 0)
      const b = Vec3.of(3, 4, 0)
      expect(Vec3.distance(a, b)).toBe(5)
    })

    test('distanceSq equals distance squared', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 6, 9)
      expect(Vec3.distanceSq(a, b)).toBeCloseTo(Vec3.distance(a, b) ** 2)
    })
  })

  describe('equals', () => {
    test('identical vectors are equal', () => {
      const v = Vec3.of(1, 2, 3)
      expect(Vec3.equals(v, v)).toBe(true)
    })

    test('different vectors are not equal', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(1, 2, 4)
      expect(Vec3.equals(a, b)).toBe(false)
    })

    test('approximately equal vectors pass with default tolerance', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(1 + 1e-13, 2 - 1e-13, 3)
      expect(Vec3.equals(a, b)).toBe(true)
    })

    test('approximately different vectors fail with default tolerance', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(1 + 1e-6, 2, 3)
      expect(Vec3.equals(a, b)).toBe(false)
    })

    test('custom tolerance works', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(1.0001, 2, 3)
      expect(Vec3.equals(a, b, 1e-3)).toBe(true)
      expect(Vec3.equals(a, b, 1e-6)).toBe(false)
    })
  })

  describe('toArray/fromArray', () => {
    test('toArray extracts components', () => {
      expect(Vec3.toArray(Vec3.of(1, 2, 3))).toEqual([1, 2, 3])
    })

    test('fromArray creates vector', () => {
      expect(Vec3.fromArray([1, 2, 3])).toEqual(Vec3.of(1, 2, 3))
    })

    test('round-trip preserves values', () => {
      const v = Vec3.of(1.5, -2.25, 3.75)
      expect(Vec3.fromArray(Vec3.toArray(v))).toEqual(v)
    })
  })

  describe('map', () => {
    test('map applies function to each component', () => {
      expect(Vec3.map(Vec3.of(1, 2, 3), x => x * 2)).toEqual(Vec3.of(2, 4, 6))
    })

    test('map with sqrt', () => {
      expect(Vec3.map(Vec3.of(4, 9, 16), Math.sqrt)).toEqual(Vec3.of(2, 3, 4))
    })

    test('map can change all components', () => {
      expect(Vec3.map(Vec3.of(1, 2, 3), () => 42)).toEqual(Vec3.of(42, 42, 42))
    })
  })

  describe('sum', () => {
    test('sum adds all components', () => {
      expect(Vec3.sum(Vec3.of(1, 2, 3))).toBe(6)
    })

    test('sum of zero vector is 0', () => {
      expect(Vec3.sum(Vec3.zero())).toBe(0)
    })
  })

  describe('abs', () => {
    test('abs makes all components positive', () => {
      expect(Vec3.abs(Vec3.of(-1, 2, -3))).toEqual(Vec3.of(1, 2, 3))
    })

    test('abs of positive vector is unchanged', () => {
      expect(Vec3.abs(Vec3.of(1, 2, 3))).toEqual(Vec3.of(1, 2, 3))
    })
  })

  describe('numeric edge cases', () => {
    test('handles very large values', () => {
      const v = Vec3.of(1e10, 1e10, 1e10)
      expect(Vec3.length(v)).toBeCloseTo(Math.sqrt(3) * 1e10)
    })

    test('handles very small values', () => {
      const v = Vec3.of(1e-10, 2e-10, 3e-10)
      expect(Vec3.length(v)).toBeCloseTo(Math.sqrt(14) * 1e-10)
    })

    test('handles infinity', () => {
      const v = Vec3.of(Infinity, 0, 0)
      expect(Vec3.length(v)).toBe(Infinity)
    })

    test('handles NaN', () => {
      const v = Vec3.of(NaN, 1, 1)
      expect(Vec3.length(v)).toBeNaN()
    })
  })

  describe('geometric identities', () => {
    test('triple product a·(b×c) is scalar', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.of(7, 8, 9)
      const scalarTriple = Vec3.dot(a, Vec3.cross(b, c))
      expect(typeof scalarTriple).toBe('number')
    })

    test('parallelogram area via cross product', () => {
      const a = Vec3.of(1, 0, 0)
      const b = Vec3.of(0, 1, 0)
      const area = Vec3.length(Vec3.cross(a, b))
      expect(area).toBeCloseTo(1)
    })

    test('triangle area via cross product', () => {
      // Triangle with vertices at origin, (1,0,0), (0,1,0)
      // Area = 0.5 * |cross product|
      const v1 = Vec3.of(1, 0, 0)
      const v2 = Vec3.of(0, 1, 0)
      const area = Vec3.length(Vec3.cross(v1, v2)) / 2
      expect(area).toBeCloseTo(0.5)
    })

    test('Cauchy-Schwarz inequality: |a·b| <= |a||b|', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Math.abs(Vec3.dot(a, b))).toBeLessThanOrEqual(Vec3.length(a) * Vec3.length(b))
    })

    test('Triangle inequality: |a+b| <= |a| + |b|', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.length(Vec3.add(a, b))).toBeLessThanOrEqual(Vec3.length(a) + Vec3.length(b))
    })

    test('Parallelogram law: |a+b|² + |a-b|² = 2|a|² + 2|b|²', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const lhs = Vec3.lengthSq(Vec3.add(a, b)) + Vec3.lengthSq(Vec3.sub(a, b))
      const rhs = 2 * Vec3.lengthSq(a) + 2 * Vec3.lengthSq(b)
      expect(lhs).toBeCloseTo(rhs)
    })
  })

  describe('consistency checks', () => {
    test('scale then dot equals scaled dot', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const s = 2.5
      expect(Vec3.dot(Vec3.scale(a, s), b)).toBeCloseTo(s * Vec3.dot(a, b))
    })

    test('sub then add returns original', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      expect(Vec3.add(Vec3.sub(a, b), b)).toEqual(a)
    })

    test('normalize then scale returns original length', () => {
      const v = Vec3.of(3, 4, 0)
      const len = Vec3.length(v)
      const n = Vec3.normalize(v)
      expect(Vec3.length(Vec3.scale(n, len))).toBeCloseTo(len)
    })

    test('cross product expansion: (a+b)×c = a×c + b×c', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.of(7, 8, 9)
      const lhs = Vec3.cross(Vec3.add(a, b), c)
      const rhs = Vec3.add(Vec3.cross(a, c), Vec3.cross(b, c))
      expect(lhs).toEqual(rhs)
    })

    test('Jacobi identity: a×(b×c) + b×(c×a) + c×(a×b) = 0', () => {
      const a = Vec3.of(1, 2, 3)
      const b = Vec3.of(4, 5, 6)
      const c = Vec3.of(7, 8, 9)
      const sum = Vec3.add(
        Vec3.cross(a, Vec3.cross(b, c)),
        Vec3.add(
          Vec3.cross(b, Vec3.cross(c, a)),
          Vec3.cross(c, Vec3.cross(a, b))
        )
      )
      expect(Vec3.length(sum)).toBeLessThan(1e-10)
    })
  })
})
