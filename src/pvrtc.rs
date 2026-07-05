//! Software decoder for PVRTC1 4bpp, used as a fallback on GPUs that don't expose
//! `VK_IMG_format_pvrtc` (i.e. essentially everything except Apple Silicon/PowerVR). Written
//! from scratch against the public PVRTC1 block layout (bilinear-interpolated colour pairs +
//! per-pixel 2-bit modulation, blocks addressed in Morton/Z-order); no third-party decoder crate.
//!
//! Each 8-byte block covers a 4x4 texel area but its colours bleed into the *surrounding* 4x4
//! areas too: a texel's final colour bilinearly blends the colour pairs of the 4 blocks whose
/// centers are nearest it, then mixes between the blended colour A and colour B using that
/// texel's own modulation weight.
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Copy, Default)]
struct Rgba {
    r: i32,
    g: i32,
    b: i32,
    a: i32,
}

#[derive(Clone, Copy, Default)]
struct Block {
    color_a: Rgba,
    color_b: Rgba,
    /// Per-texel weight in [0, 8] (i.e. eighths) used to blend color_a -> color_b.
    weights: [u8; 16],
    /// Bit i set means texel i is fully transparent (punch-through alpha).
    punch_through: u16,
}

const STANDARD_WEIGHTS: [u8; 4] = [0, 3, 5, 8];
const PUNCH_THROUGH_WEIGHTS: [u8; 4] = [0, 4, 4, 8];

/// Expands a 5-bit or 4-bit channel to 8 bits by replicating the top bits into the low bits.
fn expand5(v: u32) -> i32 {
    ((v << 3) | (v >> 2)) as i32
}
fn expand4(v: u32) -> i32 {
    ((v << 4) | v) as i32
}

/// Unpacks the two 16-bit colour words (bytes 4..8 of the block, little-endian) into colour A
/// and colour B. Each colour is either opaque RGB555 (top bit set) or translucent ARGB4443
/// (top bit clear, 3-bit alpha promoted to 4 bits by the caller via `PUNCH_THROUGH`/weight logic).
fn decode_colors(block_bytes: &[u8]) -> (Rgba, Rgba) {
    let ca = u16::from_le_bytes([block_bytes[4], block_bytes[5]]) as u32;
    let cb = u16::from_le_bytes([block_bytes[6], block_bytes[7]]) as u32;

    let color_a = if ca & 0x8000 != 0 {
        // Opaque RGB555: 0 1RRRRR GGGGG BBBBB
        Rgba {
            r: expand5((ca >> 10) & 0x1f),
            g: expand5((ca >> 5) & 0x1f),
            b: expand5(((ca & 0x1e) | ((ca >> 4) & 1)) & 0x1f),
            a: 255,
        }
    } else {
        // Translucent ARGB3443: 0 0AAA RRRR GGGG BBB(x2)
        Rgba {
            r: expand4(((ca >> 7) & 0x1e) | ((ca >> 11) & 1)),
            g: expand4(((ca >> 3) & 0x1e) | ((ca >> 7) & 1)),
            b: expand4(((ca << 1) & 0x1c) | ((ca >> 2) & 3)),
            a: (((ca >> 11) & 0xe) as i32) * 255 / 14,
        }
    };
    let color_b = if cb & 0x8000 != 0 {
        // Opaque RGB555: 1 1RRRRR GGGGG BBBBB
        Rgba {
            r: expand5((cb >> 10) & 0x1f),
            g: expand5((cb >> 5) & 0x1f),
            b: expand5(cb & 0x1f),
            a: 255,
        }
    } else {
        Rgba {
            r: expand4(((cb >> 7) & 0x1e) | ((cb >> 11) & 1)),
            g: expand4(((cb >> 3) & 0x1e) | ((cb >> 7) & 1)),
            b: expand4(((cb << 1) & 0x1e) | ((cb >> 3) & 1)),
            a: (((cb >> 11) & 0xe) as i32) * 255 / 14,
        }
    };
    (color_a, color_b)
}

/// Unpacks the 32-bit modulation word (bytes 0..4) into 16 per-texel blend weights (raster
/// order within the block) plus a punch-through-alpha bitmask.
fn decode_weights(block_bytes: &[u8]) -> ([u8; 16], u16) {
    let hard_mode = block_bytes[4] & 1 != 0;
    let mut bits = u32::from_le_bytes(block_bytes[0..4].try_into().unwrap());
    let mut weights = [0u8; 16];
    let mut punch_through = 0u16;
    for (i, weight) in weights.iter_mut().enumerate() {
        let code = (bits & 3) as usize;
        if hard_mode {
            *weight = PUNCH_THROUGH_WEIGHTS[code];
            if code == 2 {
                punch_through |= 1 << i;
            }
        } else {
            *weight = STANDARD_WEIGHTS[code];
        }
        bits >>= 2;
    }
    (weights, punch_through)
}

fn lerp8(a: i32, b: i32, weight: u8) -> i32 {
    (a * (8 - weight as i32) + b * weight as i32) / 8
}

/// Z-order (Morton) index used by PVRTC to address blocks in a cache-friendlier order than
/// raster scanning; `side` is the shorter of the block-grid's two dimensions (both must be
/// powers of two, which every PVRTC texture satisfies).
fn morton_index(x: usize, y: usize, side: usize) -> usize {
    let mut offset = 0usize;
    let mut shift = 0usize;
    let mut mask = 1usize;
    while mask < side {
        offset |= ((y & mask) | ((x & mask) << 1)) << shift;
        mask <<= 1;
        shift += 1;
    }
    offset | ((x | y) >> shift) << (shift * 2)
}

/// Decodes a headerless PVRTC1 4bpp mip level (mip 0 only, matching this project's texture
/// pipeline) into tightly packed RGBA8. `width`/`height` must each be a power of two and at
/// least 8 (PVRTC's minimum block-grid dimension).
pub fn decode_4bpp(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let side = blocks_x.min(blocks_y);
    let num_blocks = blocks_x * blocks_y;
    assert!(data.len() >= num_blocks * 8, "PVRTC data too short");

    let blocks: Vec<Block> = (0..num_blocks)
        .map(|morton| {
            let bytes = &data[morton * 8..morton * 8 + 8];
            let (color_a, color_b) = decode_colors(bytes);
            let (weights, punch_through) = decode_weights(bytes);
            Block {
                color_a,
                color_b,
                weights,
                punch_through,
            }
        })
        .collect();
    let block_at = |bx: usize, by: usize| &blocks[morton_index(bx, by, side)];

    let mut out = vec![0u8; width * height * 4];
    for by in 0..blocks_y {
        let neighbor_y = [
            if by == 0 { blocks_y - 1 } else { by - 1 },
            by,
            if by + 1 == blocks_y { 0 } else { by + 1 },
        ];
        for bx in 0..blocks_x {
            let neighbor_x = [
                if bx == 0 { blocks_x - 1 } else { bx - 1 },
                bx,
                if bx + 1 == blocks_x { 0 } else { bx + 1 },
            ];
            let this_block = block_at(bx, by);

            for local_y in 0..4usize {
                // Texels in the top half of the block lean on the block above; bottom half
                // leans on the block below (and symmetrically for x). Weight pairs (3,1)/(1,3)
                // approximate the standard bilinear taps used by the reference algorithm.
                let (near_y, y_weights) = if local_y < 2 {
                    (neighbor_y[0], [3 - local_y as i32, 1 + local_y as i32])
                } else {
                    (
                        neighbor_y[2],
                        [3 - (3 - local_y as i32), 1 + (3 - local_y as i32)],
                    )
                };
                for local_x in 0..4usize {
                    let (near_x, x_weights) = if local_x < 2 {
                        (neighbor_x[0], [3 - local_x as i32, 1 + local_x as i32])
                    } else {
                        (
                            neighbor_x[2],
                            [3 - (3 - local_x as i32), 1 + (3 - local_x as i32)],
                        )
                    };

                    let corners = [
                        (block_at(bx, by), x_weights[1] * y_weights[1]),
                        (block_at(near_x, by), x_weights[0] * y_weights[1]),
                        (block_at(bx, near_y), x_weights[1] * y_weights[0]),
                        (block_at(near_x, near_y), x_weights[0] * y_weights[0]),
                    ];

                    let mut a = Rgba::default();
                    let mut b = Rgba::default();
                    for (blk, w) in corners {
                        a.r += blk.color_a.r * w;
                        a.g += blk.color_a.g * w;
                        a.b += blk.color_a.b * w;
                        a.a += blk.color_a.a * w;
                        b.r += blk.color_b.r * w;
                        b.g += blk.color_b.g * w;
                        b.b += blk.color_b.b * w;
                        b.a += blk.color_b.a * w;
                    }
                    // Corner weights sum to 16 (4x4 bilinear taps), so divide by 16.
                    for c in [&mut a, &mut b] {
                        c.r /= 16;
                        c.g /= 16;
                        c.b /= 16;
                        c.a /= 16;
                    }

                    let texel_index = local_y * 4 + local_x;
                    let weight = this_block.weights[texel_index];
                    let transparent = this_block.punch_through & (1 << texel_index) != 0;
                    let px = [
                        lerp8(a.r, b.r, weight) as u8,
                        lerp8(a.g, b.g, weight) as u8,
                        lerp8(a.b, b.b, weight) as u8,
                        if transparent {
                            0
                        } else {
                            lerp8(a.a, b.a, weight) as u8
                        },
                    ];

                    let x = bx * 4 + local_x;
                    let y = by * 4 + local_y;
                    if x < width && y < height {
                        let out_offset = (y * width + x) * 4;
                        out[out_offset..out_offset + 4].copy_from_slice(&px);
                    }
                }
            }
        }
    }
    out
}
