export function hasAtLeastOneElement<T>(array: T[]): array is [T, ...T[]] {
  return array.length > 0
}

export function hasExactlyOneElement<T>(array: T[]): array is [T] {
  return array.length === 1
}

export function hasExactlyTwoElements<T>(array: T[]): array is [T, T] {
  return array.length === 2
}

export function hasExactlyFourElements<T>(array: T[]): array is [T, T, T, T] {
  return array.length === 4
}
