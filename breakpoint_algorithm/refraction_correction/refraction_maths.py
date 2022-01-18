import math

def refraction_maths(s1,ri,ro,n0,n1,n2):
    """The mathematical operations for distortion correction described in Fu et al.

    Parameters
    ----------
    s1: Location of point on distorted plane.
    ri: Inner radius of the tube.
    ro: Outer Radius of the tube.
    nx: Refractive index.

    Returns
    -------
    s
        The location of the point on the undistorted plane.
    """
    i=math.asin(s1/ro)
    i1=math.asin((s1/ro)*(n0/n1))
    theta=(math.pi/2)-i1+i
    s2=(ro*math.sin(i1))/math.cos(i1-i)

    if(s2*math.sin(theta)/ri<=1):
        i2=math.asin(s2*math.sin(theta)/ri)
    else:
        i2=math.pi/2
    if((n1 * math.sin(i2) / n2)>1):
        i3=math.pi/2
    else:
        i3 = math.asin(n1 * math.sin(i2) / n2)
    s=(ri*math.sin(i3))/(math.sin(theta+i2-i3))

    return s