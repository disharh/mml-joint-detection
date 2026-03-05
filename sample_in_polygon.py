## Copied from Ewoud's code (line 285 changed)

# pip install  git+https://github.com/ideasman42/isect_segments-bentley_ottmann.git
import numpy as np
from numba import njit

from numerics import *

@njit(cache=True, error_model='numpy')
def line_2points(x1, y1, x2, y2):
    # y=ax+b
    a = (y2-y1)/(x2-x1)
    b = y1-a*x1
    return a, b

@njit(cache=True)
def checksegments(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    if (x1 == x2 and y1 == y2) or (x3 == x4 and y3 == y4):
        return False
    if (x1 == x3 and y1 == y3) or (x1 == x4 and y1 == y4) or (x2 == x3 and y2 == y3) or (x2 == x4 and y2 == y4):
        return False

    # I1 = [min(x1,x2), max(x1,x2)]
    # I2 = [min(x3,x4), max(x3,x4)]
    Ia = [max(min(x1, x2), min(x3, x4)),
          min(max(x1, x2), max(x3, x4))]
    if Ia[1] < Ia[0]:
        return False

    # J1 = [min(y1,y2), max(y1,y2)]
    # J2 = [min(y3,y4), max(y3,y4)]
    Ja = [max(min(y1, y2), min(y3, y4)),
          min(max(y1, y2), max(y3, y4))]
    if Ja[1] < Ja[0]:
        return False

    if x1 == x2 or x3 == x4:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        x3, y3 = y3, x3
        x4, y4 = y4, x4
        Ia, Ja = Ja, Ia

    if x1 == x2 or x3 == x4:
        return True

    A1 = (y1 - y2) / (x1 - x2)  # Pay attention to not dividing by zero
    A2 = (y3 - y4) / (x3 - x4)  # Pay attention to not dividing by zero
    b1 = y1 - A1 * x1  # = y2-A1*x2
    b2 = y3 - A2 * x3  # = y4-A2*x4

    if (A1 == A2):
        return False  # Parallel segments

    # ya = A1 * xa + b1
    # ya = A2 * xa + b2
    # A1 * xa + b1 = A2 * xa + b2
    xa = (b2 - b1) / (A1 - A2)  # Once again, pay attention to not dividing by zero
    if xa < Ia[0] or xa > Ia[1]:
        return False  # intersection is out of bound
    else:
        return True


def intersect_4points(p1, p2, p3, p4):
    # wiki line-line inetersection
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/D
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/D
    return [px, py]


def idx_to_point(xy, idx):
    N = len(xy)
    intersect = [intersect_4points(xy[i1], xy[(i1 + 1) % N], xy[i2], xy[(i2 + 1) % N]) for i1, i2 in idx]
    return intersect

@njit(cache=True)
def find_selfint(xy):
    N = len(xy)
    idx_sortx = np.argsort(xy[:,0])
    idx_invsortx = np.zeros((N,), dtype=np.int64)
    idx_invsortx[idx_sortx] = np.arange(N)
    #sortedx = xy[:,0][idx_sortx]

    ind = []
    for i in range(N):
        a = idx_invsortx[i] # Makes x[i] = sortedx[a]
        b = idx_invsortx[(i+1)%N]
        if a<b:
            toappend = range(a+1,b)
            idx_toappend = idx_sortx[a+1:b]
        else:
            toappend = range(b+1,a)
            idx_toappend = idx_sortx[b+1:a]
        idx_toappend = np.unique(np.concatenate((idx_toappend, (idx_toappend+1)%N)))
        #toappend = range(idx_invsortx[i]+1,idx_invsortx[(i+1)%N])
        #for jj in toappend:
        for idx in idx_toappend:
            if (idx-1)%N == (i+1)%N or (idx)%N== i:
                continue
            if checksegments(xy[i], xy[(i+1)%N], xy[idx-1], xy[idx]):
                ind.append([i, (idx-1)%N])
    return ind

def fix_selfintersections3(polygon):
    """
    Fix selfintersections
    :param polygon: x and y coordinates (shape (2,N))
    :return: an array where each selfintersection has been detected and made part of the polygon
    """
    if polygon.shape[0]>polygon.shape[1]:
        raise ValueError("Probably not a good shape")
    ints = find_selfint(polygon.T)
    if len(ints) > 0:
        ints = np.unique(np.sort(ints, axis=-1), axis=0)
        points = idx_to_point(polygon.T, ints)

        ids = [i[0] for i in ints]
        sort_ind = np.argsort(ids)[::-1]
        #ids2 = [i[1] for i in ints]
        #sort_ind2 = np.argsort(ids)[::-1]
        p = np.insert(polygon, [(ids[i]+1)%len(polygon.T) for i in sort_ind], np.array(points)[sort_ind].T, axis=-1)
        #p = np.insert(polygon, [(ids[i]+1)%len(polygon.T) for i in sort_ind] + [(ids2[i]+1)%len(polygon.T) for i in sort_ind], np.array(points)[np.concatenate((sort_ind, sort_ind2))].T, axis=-1)

        return p
    else:
        return polygon


def sample_polygon_single(poly_to_sample, u):
    poly_to_sample = fix_selfintersections3(poly_to_sample)
    queue_ind = np.lexsort((poly_to_sample[1],poly_to_sample[0]), axis=0)
    points_sampled, area = sample_polygon_nb(*poly_to_sample, np.atleast_2d(u), queue_ind)
    #polya = polyarea(*poly_to_sample)
    # debug("Got area %s, vs %s", area, polya)
    #assert np.isclose(area, polya, rtol=1e-2, atol=1), f"Mistake in polgon sampling: area does not match, {area} vs {polya}"
    assert 0<area<1000, f"Mistake in polgon sampling: area not good {area}"
    x, y = points_sampled[:, 0]
    return (x, y), area


def sample_polygon(x_, y_, u_, fix_selfint=False):
    """
    Inverse-transform sample from a polygon with coordinatese x and y, at stratified coordinates u.
    Be sure to fix self-intersections first.
    :param x_: x-coordinate polygon
    :param y_: y-coordinate polygon
    :param u_: stratified coordinate
    :param return_area: whether to return the area of the polygon too
    :return: the samples of x and y coordinates within the polygon.
    """
    poly = np.array([x_,y_])
    if fix_selfint:
        poly = fix_selfintersections3(poly)
    x_, y_ = poly
    queue_ind = np.lexsort((y_,x_), axis=0)
    return sample_polygon_nb(x_, y_, np.atleast_2d(u_), queue_ind)


@njit(cache=True)
def sample_polygon_nb(x,y,ugw, queue_ind):
    N = len(x)
    active_cw = []
    active_ccw = []
    lins = [np.empty((2,2), dtype=np.float64) for ia in range(0)]
    rhos = []
    sortedx = x[queue_ind]
    #print('test')
    for i in range(len(queue_ind)-1):
        #print("start loop")
        p = queue_ind[i]
        prev_p= queue_ind[i-1]

        prev_p_on_poly = (p-1)%N
        next_p_on_poly = (p+1)%N
        acw_plus = next_p_on_poly in active_cw
        accw_min = prev_p_on_poly in active_ccw
        if acw_plus:
            ind = active_cw.index(next_p_on_poly)
            active_cw[ind] = p

        if accw_min:
            ind = active_ccw.index(prev_p_on_poly)
            active_ccw[ind] = p

        if not (acw_plus or accw_min): # new point!
            #print('new end', p, poly[:,p])
            active_cw.append(p)
            active_ccw.append(p)

        if acw_plus and accw_min:
            ind_cw = active_cw.index(p)
            ind_ccw = active_ccw.index(p)
            active_cw.pop(ind_cw)
            active_ccw.pop(ind_ccw)
            
        xleft = sortedx[i]
        #print("there")
        xright = sortedx[(i+1)%N]
        xmid = (xleft+xright)/2

        #print("start making lists")
        li = [line_2points(x[ii], y[ii], x[ii-1], y[ii-1]) for ii in active_cw] 
        li2 = [line_2points(x[ii], y[ii], x[(ii+1)%N], y[(ii+1)%N]) for ii in active_ccw]
        midpts = [line[0]*xmid+line[1] for ii, line in zip(active_cw, li)]\
                + [line[0]*xmid+line[1] for ii, line in zip(active_ccw, li2)]
        #li = []
        #midpts = []
        #for ii in active_cw:
            #line = np.array(line_2points(x[ii], y[ii], x[ii-1], y[ii-1]))
            #li.append(line)
            #midpts.append(line[0]*(x[ii]+x[ii-1])/2+line[1])
#
        #for ii in active_ccw:
            #line = np.array(line_2points(x[ii], y[ii], x[(ii+1)%N], y[(ii+1)%N]))
            #li.append(line)
            #midpts.append(line[0]*(x[ii]+x[(ii+1)%N])/2+line[1])
            
        #print("almost done making lists")
        #print(i)


        lines_ = np.array(li+li2)[np.argsort(np.array(midpts))]
        #print("here")

        lins.append(lines_)
        rho=0.
        #print("start making rhos")
        for jj in range(0,len(lines_), 2):
            al, bl = lines_[jj+0]
            au, bu = lines_[jj+1]
            #dx = (xright-xleft)
            if xleft==xright:
                area = 0.
            else:
                area = (bu-bl)*(xright-xleft) + 1/2*(au-al)*(xright**2-xleft**2)
                #area = (bu-bl)*(xright-xleft) + 1/2*(au-al)*((xright-xleft)**2)
            #print(area)
            rho += area
            """
            if area <0:
                xpl = np.linspace(xleft, xright)
                plt.figure()
                plt.plot(xpl, al*xpl+bl)
                plt.plot(xpl, au*xpl+bu)
                #plt.fill_between(xpl, al*xpl+bl, au*xpl+bu, alpha=0.5)
                plt.show()
            """
            #plt.fill_between(xpl, al*xpl+bl, au*xpl+bu, alpha=0.5)
        rhos.append(rho)

    #print('test')
    u_s = ugw
    #cs = np.zeros((len(rhos)+1))
    #print(rhos)
    cs_unnorm = np.concatenate((np.zeros((1,), dtype=np.float64), np.cumsum(np.array(rhos))))
    norm = cs_unnorm[-1]
    #logger("Cs: %s", cs)
    cs = cs_unnorm/norm
    xy_list = np.empty((len(u_s),2), dtype=np.float64)
    for ii, u in enumerate(u_s):
        p_sample = u[0]
        k = np.searchsorted(cs, p_sample) - 1
        p_intriangle = p_sample - cs[k]

        line = lins[k]
        cdfa = 1/2*(line[1::2,0]-line[::2,0]).sum()/norm
        cdfb = (line[1::2,1]-line[::2,1]).sum()/norm
        #cdfa = np.sum(np.array([1/2*(l[2]-l[0])/norm for l in line]))
        #cdfb = np.sum(np.array([(l[3]-l[1])/norm for l in line]))
        const = cdfa*sortedx[k]**2 + cdfb*sortedx[k]
        xpl = np.linspace(sortedx[k], sortedx[k+1])
        #for a, b in line:
            #plt.plot(xpl, a*xpl+b)
        discr = cdfb**2+4*cdfa*(p_intriangle+const)
        xsol = solvequadeq_single(cdfa, cdfb, -p_intriangle - const)
        #print(cdfa, cdfb, p_intriangle, const, x)
        x_filtered = [xx for xx in xsol if 2*cdfa*xx+cdfb >= 0]
        xs = x_filtered[0]

        ys_edge = np.array([a*xs+b for a, b in line])
        diffs = np.array([ys_edge[i+1]-ys_edge[i] for i in range(0,len(ys_edge), 2)])
        #y_sum = np.zeros((len(diffs) + 1))
        #np.cumsum(diffs, out=y_sum[1:])
        y_sum = np.concatenate((np.zeros((1,), dtype=np.float64), np.cumsum(diffs)))
        norm_y = y_sum[-1]
        idx_y = np.searchsorted(y_sum, norm_y*u[1])-1
        ys = ys_edge[idx_y*2] + (norm_y*u[1] - y_sum[idx_y])
        xy_list[ii] = np.array([xs,ys])

    #plt.plot(sortedx[:-1], rhos)
    #print(lins[-1])
    return xy_list.T, norm#) if return_area else np.array([x_list, y_list])

if __name__=="__main__":
    x_s = np.array([-0.0038469 , -0.0033911 , -0.00306354, -0.00287171, -0.00282547,
       -0.00293627, -0.00317442, -0.00317615, -0.00346744, -0.00392648,
       -0.00432636, -0.0044714 , -0.00505258, -0.00548901, -0.00569218,
       -0.00634186, -0.00647948, -0.00691721, -0.00726778, -0.00745164,
       -0.00782425, -0.00788636, -0.00816118, -0.0081761 , -0.00828918,
       -0.00828833, -0.00822488, -0.00820272, -0.00799479, -0.00791247,
       -0.0076312 , -0.00742199, -0.00717131, -0.00674585, -0.00665359,
       -0.00606251, -0.00589391, -0.00544489, -0.00491246, -0.00486269,
       -0.00424239, -0.00384784, -0.00369353, -0.00315213, -0.00276216,
       -0.00269691, -0.00224875, -0.00189056, -0.00174521, -0.00157706,
       -0.00130753, -0.00110424, -0.00095473, -0.00091953, -0.00083516,
       -0.00075271, -0.00070555, -0.0006846 , -0.0006816 , -0.00068874,
       -0.00069829, -0.00070211, -0.00069126, -0.0006555 , -0.00058299,
       -0.00056181, -0.00049094, -0.00035402, -0.00015046,  0.00011092,
        0.00013142,  0.00043645,  0.00085081,  0.00108413,  0.00133031,
        0.00189736,  0.00222735,  0.00254735,  0.00329766,  0.00347195,
        0.00409671,  0.00476292,  0.00500153,  0.00596806,  0.00605369,
        0.00694812,  0.00728844,  0.00795957,  0.00844026,  0.00896236,
        0.00947379,  0.00991168,  0.01035809,  0.01075725,  0.01107003,
        0.01144367,  0.01159656,  0.0119094 ,  0.01193637,  0.01206281,
        0.01207884,  0.0119568 ,  0.01188585,  0.0116985 ,  0.01130093,
        0.01125944,  0.01068218,  0.01012197,  0.0100458 ,  0.00921192,
        0.00844024,  0.00838589,  0.00752703,  0.00667916,  0.00591942,
        0.00583923,  0.00512513,  0.00442593,  0.00384694,  0.00339108,
        0.00306355,  0.00287171,  0.00282548,  0.00293623,  0.00317442,
        0.00317616,  0.00346747,  0.00392645,  0.0043264 ,  0.00447148,
        0.00505252,  0.00548901,  0.00569218,  0.00634181,  0.00647947,
        0.00691726,  0.00726777,  0.00745157,  0.00782427,  0.00788643,
        0.00816138,  0.00817624,  0.00828904,  0.00828823,  0.00822482,
        0.00820266,  0.00799473,  0.00791246,  0.00763118,  0.00742212,
        0.00717139,  0.00674575,  0.00665348,  0.00606259,  0.00589399,
        0.00544489,  0.00491249,  0.00486278,  0.00424237,  0.00384788,
        0.0036936 ,  0.00315217,  0.00276217,  0.00269688,  0.00224872,
        0.00189059,  0.00174521,  0.00157704,  0.00130753,  0.00110425,
        0.00095474,  0.00091953,  0.00083515,  0.00075271,  0.00070555,
        0.00068461,  0.0006816 ,  0.00068874,  0.00069829,  0.00070211,
        0.00069126,  0.0006555 ,  0.00058299,  0.00056181,  0.00049094,
        0.00035403,  0.00015046, -0.00011094, -0.00013144, -0.00043648,
       -0.0008508 , -0.00108415, -0.00133031, -0.00189734, -0.00222737,
       -0.00254738, -0.00329757, -0.0034719 , -0.00409674, -0.00476293,
       -0.00500151, -0.00596805, -0.00605366, -0.00694811, -0.00728845,
       -0.00795956, -0.00844032, -0.00896245, -0.00947375, -0.00991159,
       -0.01035806, -0.01075728, -0.01107009, -0.01144358, -0.01159651,
       -0.01190935, -0.01193619, -0.01206283, -0.01207889, -0.01195697,
       -0.01188591, -0.01169853, -0.01130094, -0.01125929, -0.01068225,
       -0.01012189, -0.01004586, -0.00921195, -0.00844027, -0.00838592,
       -0.00752715, -0.00667913, -0.00591942, -0.00583923, -0.00512513,
       -0.00442595, -0.0038469 ])
    y_s = np.array([ 0.00121091,  0.00120806,  0.0012219 ,  0.00124012,  0.00124887,
        0.00123191,  0.0011805 ,  0.00117984,  0.00110249,  0.0009569 ,
        0.00081078,  0.00075367,  0.00050417,  0.00029513,  0.00019153,
       -0.00016795, -0.00024986, -0.00052477, -0.00076266, -0.00089435,
       -0.00117932, -0.00122949, -0.00146441, -0.00147789, -0.00158454,
       -0.00158363, -0.0015138 , -0.00148868, -0.00124017, -0.0011366 ,
       -0.00076502, -0.00047244, -0.00010547,  0.00055698,  0.00070703,
        0.00172255,  0.00202977,  0.00288719,  0.00398306,  0.00409013,
        0.00549803,  0.00647191,  0.00687136,  0.00836829,  0.00955554,
        0.00976467,  0.01129939,  0.01268182,  0.01329518,  0.014053  ,
        0.015412  ,  0.01661507,  0.01766813,  0.01794869,  0.01870387,
        0.01964234,  0.02042419,  0.0210531 ,  0.02153117,  0.02185821,
        0.0220312 ,  0.02204434,  0.02188906,  0.0215542 ,  0.02102676,
        0.02088988,  0.02046575,  0.01974234,  0.01880425,  0.01774159,
        0.01766313,  0.01655855,  0.01520545,  0.01449915,  0.01378864,
        0.01226398,  0.01143623,  0.01066846,  0.00898413,  0.00861332,
        0.00733912,  0.00606604,  0.00562938,  0.00395511,  0.00381364,
        0.00239737,  0.00188676,  0.00092304,  0.00026706, -0.00041376,
       -0.0010489 , -0.00156753, -0.00207182, -0.00250083, -0.00282174,
       -0.00318546, -0.00332755, -0.00360269, -0.0036253 , -0.00372735,
       -0.00373936, -0.00365534, -0.0036097 , -0.00349344, -0.00326646,
       -0.00324397, -0.00294981, -0.00269363, -0.0026607 , -0.00232737,
       -0.00206003, -0.00204257, -0.00178997, -0.00158285, -0.0014329 ,
       -0.00141903, -0.00131323, -0.00124205, -0.00121091, -0.00120806,
       -0.0012219 , -0.00124012, -0.00124886, -0.00123192, -0.0011805 ,
       -0.00117984, -0.00110248, -0.00095691, -0.00081077, -0.00075364,
       -0.00050419, -0.00029512, -0.00019154,  0.00016792,  0.00024985,
        0.0005248 ,  0.00076265,  0.00089429,  0.00117933,  0.00122954,
        0.00146459,  0.00147802,  0.0015844 ,  0.00158353,  0.00151373,
        0.00148861,  0.00124011,  0.0011366 ,  0.000765  ,  0.00047263,
        0.00010559, -0.00055715, -0.00070721, -0.00172241, -0.00202962,
       -0.00288718, -0.00398298, -0.00408995, -0.00549809, -0.0064718 ,
       -0.00687118, -0.00836817, -0.00955552, -0.00976476, -0.01129949,
       -0.01268171, -0.01329517, -0.01405308, -0.01541202, -0.01661499,
       -0.01766805, -0.01794867, -0.01870392, -0.01964236, -0.02042423,
       -0.02105304, -0.02153117, -0.02185821, -0.02203118, -0.02204435,
       -0.02188906, -0.02155421, -0.02102676, -0.02088987, -0.02046574,
       -0.01974237, -0.01880427, -0.01774153, -0.01766305, -0.01655844,
       -0.01520547, -0.01449909, -0.01378864, -0.01226403, -0.01143616,
       -0.01066838, -0.00898432, -0.00861343, -0.00733906, -0.00606603,
       -0.00562942, -0.00395512, -0.00381369, -0.00239738, -0.00188676,
       -0.00092305, -0.00026698,  0.00041388,  0.00104885,  0.00156742,
        0.00207178,  0.00250086,  0.0028218 ,  0.00318538,  0.00332751,
        0.00360265,  0.00362515,  0.00372737,  0.0037394 ,  0.00365545,
        0.00360974,  0.00349346,  0.00326646,  0.00324389,  0.00294984,
        0.00269359,  0.00266073,  0.00232738,  0.00206003,  0.00204258,
        0.00179   ,  0.00158284,  0.0014329 ,  0.00141903,  0.00131322,
        0.00124205,  0.00121091])

    polygon = np.array([x_s,y_s])
    uu = np.random.uniform(0,1,size=(100000,2))
    poly = fix_selfintersections3(polygon)
    print(polygon.shape, poly.shape)
    points, area = sample_polygon(*poly[:,:], uu)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(*poly, marker='o')
    #for i in range(len(x)):
        #ax.annotate(f'{i}', (x[i], y[i]))
    ax.scatter(*points, marker='.',s=0.1)
    plt.show()


def generate_point_in_polygon(x_poly, y_poly, rg=np.random):
    from shapely.geometry import Polygon, Point
    xrange = x_poly.min(), x_poly.max()
    yrange = y_poly.min(), y_poly.max()
    polygon = Polygon(np.array([x_poly, y_poly]).T)
    while True:
        x = rg.uniform(*xrange)
        y = rg.uniform(*yrange)
        p = Point([x,y])
        if polygon.contains(p):
            return x,y