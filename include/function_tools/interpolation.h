#ifndef INTERPOLATION_H
#define INTERPOLATION_H

/** \brief Interpolate set of points which lies in input array domain
 *
 * Given a set of data points pairs `(xinp[i], yinp[i])` with domain
 * boundaries given by `xinp[0]` and `xinp[inp_size - 1]`, and a set
 * of grid points `xinterpol` contained in these boundaries, compute
 * approximately the function values `yinterpol` using interpolation
 * in a set of closest neighbouring grid points for each `xinterpol`
 *
 * For example given `x = xinterpol[i]` for some `0 <= i < out_size`
 * find the closest point to `x` in `xinp`, for instance `xinp[j]`,
 * then all points `j - neighbors / 2, ..., j + neighbors` are used
 * for polynomial interpolation in obtaining `yinterpol[i]`
 *
 * \param[in] inp_size number of input grid points in `xinp` and `yinp`
 * \param[in] neighbors number of neighbouring points to use
 * \param[in] xinp known grid points
 * \param[in] yinp known function values in grid points `xinp`
 * \param[in] out_size size of array `xinterpol` (interpolation grid)
 * \param[in] xinterpol grid points to obtain approximately function values
 * \param[out] yinterpol new function values approximated by interpolation
 */
void
lagrange(
    int    inp_size,
    int    neighbors,
    double xinp[],
    double yinp[],
    int    out_size,
    double xinterpol[],
    double yinterpol[]);

#endif
