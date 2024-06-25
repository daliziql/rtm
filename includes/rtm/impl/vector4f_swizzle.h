#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "rtm/math.h"
#include "rtm/types.h"
#include "rtm/impl/compiler_utils.h"
#include "rtm/scalarf.h"
#include "rtm/scalard.h"
#include "rtm/version.h"

RTM_IMPL_FILE_PRAGMA_PUSH

#if !defined(RTM_NO_INTRINSICS)
namespace rtm
{
	RTM_IMPL_VERSION_NAMESPACE_BEGIN

#if defined(RTM_SSE2_INTRINSICS) || defined(RTM_AVX_INTRINSICS)

#define SHUFFLE_MASK(a0,a1,b2,b3) ( (a0) | ((a1)<<2) | ((b2)<<4) | ((b3)<<6) )

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float swizzle
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4f vector_swizzle_impl(const vector4f& vec)
	{
		return _mm_shuffle_ps(vec, vec, SHUFFLE_MASK(index0, index1, index2, index3));
	}
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<0, 1, 2, 3>(const vector4f& vec)
    {
        return vec;
    }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<0, 1, 0, 1>(const vector4f& vec)
    {
        return _mm_movelh_ps(vec, vec);
    }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<2, 3, 2, 3>(const vector4f& vec)
    {
        return _mm_movehl_ps(vec, vec);
    }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<0, 0, 1, 1>(const vector4f& vec)
    {
        return _mm_unpacklo_ps(vec, vec);
    }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<2, 2, 3, 3>(const vector4f& vec)
    {
        return _mm_unpackhi_ps(vec, vec);
    }

#if defined(RTM_SSE4_INTRINSICS)
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<0, 0, 2, 2>(const vector4f& vec)
    {
        return _mm_moveldup_ps(vec);
    }
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<1, 1, 3, 3>(const vector4f& vec)
    {
        return _mm_movehdup_ps(vec);
    }
#endif

#if defined(RTM_AVX2_INTRINSICS)
	template<> RTM_FORCE_INLINE vector4f vector_swizzle_impl<0, 0, 0, 0>(const vector4f& vec)
    {
        return _mm_broadcastss_ps(vec);
    }
#endif

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float replicate
	template<int index>
	RTM_FORCE_INLINE vector4f vector_replicate_impl(const vector4f& vec)
	{
		static_assert(index >= 0 && index <= 3, "Invalid Index");
		return vector_swizzle_impl<index, index, index, index>(vec);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Float shuffle
	template<int index0, int index1, int index2, int index3>
	RTM_FORCE_INLINE vector4f vector_shuffle_impl(const vector4f& vec1, const vector4f& vec2)
	{
		static_assert(index0 >= 0 && index0 <= 3 && index1 >= 0 && index1 <= 3 && index2 >= 0 && index2 <= 3 && index3 >= 0 && index3 <= 3, "Invalid Index");
		return _mm_shuffle_ps(vec1, vec2, SHUFFLE_MASK(index0, index1, index2, index3));
	}

	// Float Shuffle specializations
	template<> RTM_FORCE_INLINE vector4f vector_shuffle_impl<0, 1, 0, 1>(const vector4f& vec1, const vector4f& vec2)
    {
        return _mm_movelh_ps(vec1, vec2);

    }
	template<> RTM_FORCE_INLINE vector4f vector_shuffle_impl<2, 3, 2, 3>(const vector4f& vec1, const vector4f& vec2)
    {
        // Note: movehl copies first from the 2nd argument
        return _mm_movehl_ps(vec2, vec1);
    }

#elif defined(RTM_NEON_INTRINSICS)
    template <int element_index>
    RTM_FORCE_INLINE vector4f vector_replicate_impl(const vector4f& vec)
    {
        return vdupq_n_f32(vgetq_lane_f32(vec, element_index));
    }

#if defined(__clang__)
    template <int x, int y, int z, int w>
    RTM_FORCE_INLINE vector4f vector_swizzle_impl(vector4f vec)
    {
        return __builtin_shufflevector(vec, vec, x, y, z, w);
    }

    template <int x, int y, int z, int w>
    RTM_FORCE_INLINE vector4f vector_shuffle_impl(vector4f vec1, vector4f vec2)
    {
        return __builtin_shufflevector(vec1, vec2, x, y, z + 4, w + 4);
    }
#else

    template<int index0, int index1, int index2, int index3>
    RTM_FORCE_INLINE vector4f vector_shuffle_impl(vector4f vec1, vector4f vec2)
    {
        static_assert(index0 <= 3 && index1 <= 3 && index2 <= 3 && index3 <= 3);

        static constexpr uint32_t control_element[8] =
        {
            0x03020100, // XM_PERMUTE_0X
            0x07060504, // XM_PERMUTE_0Y
            0x0B0A0908, // XM_PERMUTE_0Z
            0x0F0E0D0C, // XM_PERMUTE_0W
            0x13121110, // XM_PERMUTE_1X
            0x17161514, // XM_PERMUTE_1Y
            0x1B1A1918, // XM_PERMUTE_1Z
            0x1F1E1D1C, // XM_PERMUTE_1W
        };

        uint8x8x4_t tbl;
        tbl.val[0] = vget_low_f32(vec1);
        tbl.val[1] = vget_high_f32(vec1);
        tbl.val[2] = vget_low_f32(vec2);
        tbl.val[3] = vget_high_f32(vec2);

        uint32x2_t idx = vcreate_u32(static_cast<uint64_t>(control_element[index0]) | (static_cast<uint64_t>(control_element[index1]) << 32));
        const uint8x8_t rL = vtbl4_u8(tbl, idx);

        idx = vcreate_u32(static_cast<uint64_t>(control_element[index2 + 4]) | (static_cast<uint64_t>(control_element[index3 + 4]) << 32));
        const uint8x8_t rH = vtbl4_u8(tbl, idx);

        return vcombine_f32(rL, rH);
    }

	template<int index0, int index1, int index2, int index3>
    RTM_FORCE_INLINE vector4f vector_swizzle_impl(vector4f vec)
    {
        static_assert((index0 < 4) && (index1 < 4) && (index2 < 4) && (index3 < 4));
        static constexpr uint32_t control_element[4] =
        {
            0x03020100, // XM_SWIZZLE_X
            0x07060504, // XM_SWIZZLE_Y
            0x0B0A0908, // XM_SWIZZLE_Z
            0x0F0E0D0C, // XM_SWIZZLE_W
        };

        uint8x8x2_t tbl;
        tbl.val[0] = vget_low_f32(vec);
        tbl.val[1] = vget_high_f32(vec);

        uint32x2_t idx = vcreate_u32(static_cast<uint64_t>(control_element[index0]) | (static_cast<uint64_t>(control_element[index1]) << 32));
        const uint8x8_t rL = vtbl2_u8(tbl, idx);

        idx = vcreate_u32(static_cast<uint64_t>(control_element[index2]) | (static_cast<uint64_t>(control_element[index3]) << 32));
        const uint8x8_t rH = vtbl2_u8(tbl, idx);

        return vcombine_f32(rL, rH);
    }
#endif
#else
#pragma error("vector swizzle not implement here!");
#endif

#define VECTOR_REPLICATE( vec, element_index )	        vector_replicate_impl<element_index>(vec)
#define VECTOR_SWIZZLE( vec, x, y, z, w )		        vector_swizzle_impl<x,y,z,w>( vec )
#define VECTOR_SHUFFLE( vec1, vec2, x, y, z, w )		vector_shuffle_impl<x,y,z,w>( vec1, vec2 )

RTM_IMPL_VERSION_NAMESPACE_END
}
#endif
RTM_IMPL_FILE_PRAGMA_POP

