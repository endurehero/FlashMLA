#pragma once

template <int kBlockN, int kHeadDim, typename Element>
struct SmemTransposeFp8_64x64 {
    static_assert(sizeof(Element) == 1);
    static_assert((kBlockN % 64 == 0) && (kHeadDim % 64 == 0));

    using SmemLayoutK = decltype(tile_to_shape(
            GMMA::Layout_K_SW64_Atom<Element>{},
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(composition(
            SmemLayoutK{},
            Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<_1, Int<kBlockN>>>{}));
    using TransposeShapeAtomV = Shape<_64, _64>;
    
    // for fp8 in-kernel transpose -- src layout
    using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
    using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
    using FactoringShapeV = decltype(make_shape(SmemShapeLDSM{}, shape<1>(SmemLayoutDivideV{}), shape<2>(SmemLayoutDivideV{})));
    using SmemLayoutTransposeV = decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

    // For fp8, this is the memory transpose.
    using SmemLayoutAtomVt = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}));
    using SmemLayoutVt = decltype(tile_to_shape(
        SmemLayoutAtomVt{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // for fp8 in-kernel transpose -- dst layout
    using SmemLayoutVtTrans = decltype(composition(
        SmemLayoutVt{}, make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1>{})));
    using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
    using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
    using FactoringShapeVt = decltype(make_shape(SmemShapeSTSM{}, shape<1>(SmemLayoutDivideVt{}), shape<2>(SmemLayoutDivideVt{})));
    using SmemLayoutTransposeVt = decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));


    using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
    using ldsm_value_shape = Shape<_2, _8, _2, _1>;
    using ldsm_value_stride = Stride<_2, _4, _1, _0>;
    using TiledCopyLDSM = decltype(make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
                                                    Layout<ldsm_value_shape, ldsm_value_stride>{}));
    TiledCopyLDSM tiled_copy_ldsm;

    using stsm_thread_shape = Shape<_4, _1, _8, _4>;
    // using stsm_thread_stride = Stride<_1, _0, _4, _32>;
    using stsm_value_shape = Shape<_4, _4, _1, _2>;
    using stsm_value_stride = Stride<_1, _8, _0, _4>;

    using TiledCopySTSM = decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<stsm_thread_shape>{},
                                                    Layout<stsm_value_shape, stsm_value_stride>{}));
    TiledCopySTSM tiled_copy_stsm;

    template <class SmemTensor, class SmemTensorOut>
    CUTLASS_DEVICE void transpose(SmemTensor &&s_in, SmemTensorOut &&s_out) {
        using namespace cute;

        auto tid = threadIdx.x;
        auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
        auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

        auto tXsX = thr_copy_ldsm.partition_S(s_in);
        auto tXrX = make_tensor<Element>(shape(tXsX));
        auto tXsX_out = thr_copy_stsm.partition_D(s_out);

        cute::copy(tiled_copy_ldsm, tXsX, tXrX);

        auto data = tXrX.data();
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size(tXrX); n += 8) {
        uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
        auto upper = data_32bit[0];
        auto lower = data_32bit[1];
        data_32bit[0] = __byte_perm(upper, lower, 0x6420);
        data_32bit[1] = __byte_perm(upper, lower, 0x7531);
        }

        cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
    }
};

