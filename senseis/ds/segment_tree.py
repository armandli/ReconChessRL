import math
import operator

def nearest_power(n):
  v = n - 1
  v |= v >> 1
  v |= v >> 2
  v |= v >> 4
  v |= v >> 8
  v |= v >> 16
  v |= v >> 32
  v += 1
  return v

class SegmentTree1DV1:
  #NOTE: do not use the constructor, use the build method to make segment tree
  def __init__(self, t, n, op, dv):
    self.t = t   # segment tree represented as array
    self.n = n   # array length
    self.op = op # operation
    self.dv = dv # default value

  def __len__(self):
    return self.n

  def segment_op(self, l, r):
    def recursion(v, tl, tr, l, r):
      if l >= r:
        return self.dv
      if l == tl and r == tr:
        return self.t[v]
      tm = math.floor((tr - tl) / 2) + tl
      return self.op(
        recursion(v * 2 + 1, tl, tm, l, min(r, tm)),
        recursion(v * 2 + 2, tm, tr, max(tm, l), r)
      )
    return recursion(0, 0, self.n, l, r)

  def __getitem__(self, pos):
    def recursion(v, tl, tr, pos):
      if tl + 1 == tr:
        return self.t[v]
      tm = math.floor((tr - tl) / 2) + tl
      if pos < tm:
        return recursion(v * 2 + 1, tl, tm, pos)
      else:
        return recursion(v * 2 + 2, tm, tr, pos)
    return recursion(0, 0, self.n, pos)

  def __setitem__(self, pos, nv):
    def recursion(v, tl, tr, pos, nv):
      if tl + 1 == tr:
        self.t[v] = nv
        return
      tm = math.floor((tr - tl) / 2) + tl
      if pos < tm:
        recursion(v * 2 + 1, tl, tm, pos, nv)
      else:
        recursion(v * 2 + 2, tm, tr, pos, nv)
      self.t[v] = self.op(self.t[v * 2 + 1], self.t[v * 2 + 2])
    recursion(0, 0, self.n, pos, nv)

class SumTree1DV1(SegmentTree1DV1):
  def __init__(self, t, n, op, dv):
    super(SumTree1DV1, self).__init__(t, n, op, dv)

  def sum(self, l=0, r=None):
    if r is None:
      r = self.n
    return self.segment_op(l, r)

  # find the highest index i s.t. sum(a[0], .., a[i-1]) <= prefix_sum
  def prefix_sum_idx(self, prefix_sum):
    def recursion(v, tl, tr, ps):
      if tl + 1 == tr:
        return tl
      tm = math.floor((tr - tl) / 2) + tl
      lsum = self.t[v * 2 + 1]
      if lsum <= ps:
        return recursion(v * 2 + 2, tm, tr, ps - lsum)
      else:
        return recursion(v * 2 + 1, tl, tm, ps)
    if prefix_sum >= self.sum():
      return self.n - 1
    return recursion(0, 0, self.n, prefix_sum)

  @staticmethod
  def build(a):
    n = nearest_power(len(a))
    dv = 0
    op = operator.add
    t = [dv for _ in range(n * 2)]
    def recursion(t, a, v, l, r, op):
      if l + 1 == r:
        t[v] = a[l]
        return
      m = math.floor((r - l) / 2) + l
      recursion(t, a, v * 2 + 1, l, m, op)
      recursion(t, a, v * 2 + 2, m, r, op)
      t[v] = op(t[v * 2 + 1], t[v * 2 + 2])
    recursion(t, a, 0, 0, len(a), op)
    return SumTree1DV1(t, len(a), op, dv)


class MinTree1DV1(SegmentTree1DV1):
  def __init__(self, t, n, op, dv):
    super(MinTree1DV1, self).__init__(t, n, op, dv)

  def min(self, l=0, r=None):
    if r is None:
      r = self.n
    return self.segment_op(l, r)

  @staticmethod
  def build(a):
    n = nearest_power(len(a))
    dv = math.inf
    op = min
    t = [dv for _ in range(n * 2)]
    def recursion(t, a, v, l, r, op):
      if l + 1 == r:
        t[v] = a[l]
        return
      m = math.floor((r - l) / 2) + l
      recursion(t, a, v * 2 + 1, l, m, op)
      recursion(t, a, v * 2 + 2, m, r, op)
      t[v] = op(t[v * 2 + 1], t[v * 2 + 2])
    recursion(t, a, 0, 0, len(a), op)
    return MinTree1DV1(t, len(a), op, dv)

class MaxTree1DV1(SegmentTree1DV1):
  def __init__(self, t, n, op, dv):
    super(MaxTree1DV1, self).__init__(t, n, op, dv)

  def max(self, l=0, r=None):
    if r is None:
      r = self.n
    return self.segment_op(l, r)

  @staticmethod
  def build(a):
    n = nearest_power(len(a))
    dv = -math.inf
    op = max
    t = [dv for _ in range(n * 2)]
    def recursion(t, a, v, l, r, op):
      if l + 1 == r:
        t[v] = a[l]
        return
      m = math.floor((r - l) / 2) + l
      recursion(t, a, v * 2 + 1, l, m, op)
      recursion(t, a, v * 2 + 2, m, r, op)
      t[v] = op(t[v * 2 + 1], t[v * 2 + 2])
    recursion(t, a, 0, 0, len(a), op)
    return MaxTree1DV1(t, len(a), op, dv)