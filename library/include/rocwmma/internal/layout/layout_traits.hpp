/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_LAYOUT_TRAITS_HPP
#define ROCWMMA_LAYOUT_TRAITS_HPP

#include "data_layout_traits.hpp"
#include "layout_traits_impl.hpp"
#include "matrix_layout_traits.hpp"
#include "register_layout_traits.hpp"

namespace rocwmma
{
    /*! \class is_layout_same
    *  \brief  Compares layout types are the same, or are equivalent. Similar to is_same,
    * however layouts can have an equivalency with small variations input parameters such that they
    * are still technically the same. This should be used when comparing any layout types:
    * DataLayout, MatrixLayout and RegisterLayout
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_same : public LayoutTraits_impl::is_layout_same<LhsLayout, RhsLayout>
    {
    };

    /*! \class is_layout_orthogonal
    *  \brief  Compares layout types if they are orthogonal with each other.
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_orthogonal
        : public LayoutTraits_impl::is_layout_orthogonal<LhsLayout, RhsLayout>
    {
    };

    /*! \class orthogonal_layout
    *  \brief  Transforms the layout type into its orthogonal layout.
    *  @tparam Layout the layout to transpose from
    */
    template <typename Layout>
    struct orthogonal_layout : public LayoutTraits_impl::orthogonal_layout<Layout>
    {
    };

    /*! \class is_layout_same_v
    *  \brief  Evaluates is_layout_same
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    constexpr static inline bool is_layout_same_v = is_layout_same<LhsLayout, RhsLayout>::value;

    /*! \class is_layout_orthogonal
    *  \brief  Evaluates is_layout_orthogonal
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    constexpr static inline bool is_layout_orthogonal_v
        = is_layout_orthogonal<LhsLayout, RhsLayout>::value;

    /*! \class layout_transpose_t
    *  \brief  Transforms the layout type into its orthogonal layout.
    *  @tparam Layout the layout to transpose from
    */
    template <typename Layout>
    using orthogonal_layout_t = typename orthogonal_layout<Layout>::type;

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRAITS_HPP
