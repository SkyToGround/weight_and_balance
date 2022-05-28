import cython


def on_segment(p, q, r) -> bool:
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False

def orientation(p, q, r) -> cython.int:
    val: cython.float = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0: return 0
    return 1 if val > 0 else -1


def intersects(seg1, seg2) -> bool:
    p1, q1 = seg1
    p2, q2 = seg2

    o1: cython.int = orientation(p1, q1, p2)
    o2: cython.int = orientation(p1, q1, q2)
    o3: cython.int = orientation(p2, q2, p1)
    o4: cython.int = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2) : return True
    if o2 == 0 and on_segment(p1, q1, q2) : return True
    if o3 == 0 and on_segment(p2, q2, p1) : return True
    if o4 == 0 and on_segment(p2, q2, q1) : return True
    return False

def line_intersection(seg1, seg2) -> bool:
    p1, p2 = seg1
    p3, p4 = seg2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    if denom==0: return False
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if (px - x1) * (px - x2) < 0 and (py - y1) * (py - y2) < 0 \
      and (px - x3) * (px - x4) < 0 and (py - y3) * (py - y4) < 0:
        return True
    return False