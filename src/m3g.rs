//! Parser for the JSR-184 (M3G 1.0) binary file format.
//! Spec: JSR 184 "Mobile 3D Graphics API for J2ME", Appendix on the binary file format.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::pvrtc;

const MAGIC: [u8; 12] = *b"\xabJSR184\xbb\r\n\x1a\n";

pub enum ImageError {
    UnknownFormat(u8),
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjRef(pub u32);

impl ObjRef {
    pub const NULL: ObjRef = ObjRef(0);
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug, Clone, Default)]
pub enum ImageBytes {
    #[default]
    Empty,
    Borrowed(Rc<[u8]>, Range<usize>),
}

impl ImageBytes {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Borrowed(buf, range) => &buf[range.clone()],
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Object3DBase {
    pub user_id: u32,
    pub animation_tracks: Vec<ObjRef>,
    pub user_parameters: Vec<(u32, ImageBytes)>,
}

#[derive(Debug, Default, Clone)]
pub struct NodeBase {
    pub base: Object3DBase,
    pub enable_rendering: bool,
    pub enable_picking: bool,
    pub alpha_factor: u8,
    pub scope: u32,
    pub translation: [f32; 3],
    pub orientation_angle: f32,
    pub orientation_axis: [f32; 3],
    pub scale: [f32; 3],
    pub general_transform: Option<[f32; 16]>,
    pub alignment: Option<Alignment>,
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub z_target: u8,
    pub y_target: u8,
    pub z_reference: ObjRef,
    pub y_reference: ObjRef,
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: (u8, u8),
    pub has_external_references: bool,
    pub total_file_size: u32,
    pub approximate_content_size: u32,
    pub authoring_field: String,
}

#[derive(Debug, Clone)]
pub struct AnimationController {
    pub base: Object3DBase,
    pub speed: f32,
    pub weight: f32,
    pub active_interval_start: u32,
    pub active_interval_end: u32,
    pub reference_sequence_time: f32,
    pub reference_world_time: i32,
}

#[derive(Debug, Clone)]
pub struct AnimationTrack {
    pub base: Object3DBase,
    pub keyframe_sequence: ObjRef,
    pub animation_controller: ObjRef,
    pub property_id: u32,
}

#[derive(Debug, Clone)]
pub enum KeyframeData {
    Float(Vec<(u32, Vec<f32>)>),
    Quantized {
        bias: Vec<f32>,
        scale: Vec<f32>,
        // (time, quantized components)
        frames: Vec<(u32, Vec<i32>)>,
    },
}

#[derive(Debug, Clone)]
pub struct KeyframeSequence {
    pub base: Object3DBase,
    pub interpolation: u8,
    pub repeat_mode: u8,
    pub duration: u32,
    pub valid_range_first: u32,
    pub valid_range_last: u32,
    pub component_count: u32,
    pub keyframe_count: u32,
    pub data: KeyframeData,
}

#[derive(Debug, Clone)]
pub struct Image2D {
    pub base: Object3DBase,
    /// JSR184 pixel format: 96=ALPHA, 97=LUMINANCE, 98=LUMINANCE_ALPHA, 99=RGB, 100=RGBA.
    pub format: u8,
    pub is_mutable: bool,
    pub width: u32,
    pub height: u32,
    pub palette: ImageBytes,
    pub pixels: ImageBytes,
}

impl Image2D {
    /// Expands to tightly-packed RGBA8 regardless of source format.
    ///
    /// Formats 96-100 are stock JSR184. This EA engine additionally stores GPU-compressed
    /// images (observed 124/125 = PVRTC 4bpp variants, sized as a headerless mip chain down
    /// to 8x8 — verified against `m3g::Image2D::commit` in the game binary and exact byte
    /// counts of the shipped textures). Only mip 0 is decoded here.
    pub fn to_rgba8(&self) -> Result<Vec<u8>, ImageError> {
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;
        let bpp = match self.format {
            96 | 97 => 1,
            98 => 2,
            99 => 3,
            100 => 4,
            124 | 125 => {
                let mip0_len = (w.max(8) * h.max(8)) / 2;
                return Ok(pvrtc::decode_4bpp(
                    w,
                    h,
                    &self.pixels.as_slice()[..mip0_len],
                ));
            }
            other => {
                return Err(ImageError::UnknownFormat(other));
            }
        };
        let mut out = Vec::with_capacity(n * 4);
        let palette = self.palette.as_slice();
        let pixels = self.pixels.as_slice();
        // Palettized when a palette is present: each pixel byte indexes the palette.
        if !palette.is_empty() {
            for i in 0..n {
                let idx = pixels[i] as usize * bpp;
                let e = &palette[idx..idx + bpp];
                out.extend_from_slice(&expand_px(self.format, e));
            }
        } else {
            for i in 0..n {
                let e = &pixels[i * bpp..i * bpp + bpp];
                out.extend_from_slice(&expand_px(self.format, e));
            }
        }

        Ok(out)
    }
}

fn expand_px(format: u8, e: &[u8]) -> [u8; 4] {
    match format {
        96 => [255, 255, 255, e[0]],
        97 => [e[0], e[0], e[0], 255],
        98 => [e[0], e[0], e[0], e[1]],
        99 => [e[0], e[1], e[2], 255],
        100 => [e[0], e[1], e[2], e[3]],
        _ => unreachable!(),
    }
}

#[derive(Debug, Clone)]
pub struct Appearance {
    pub base: Object3DBase,
    pub layer: i8,
    pub compositing_mode: ObjRef,
    pub fog: ObjRef,
    pub polygon_mode: ObjRef,
    pub material: ObjRef,
    pub textures: Vec<ObjRef>,
}

#[derive(Debug, Clone)]
pub struct Group {
    pub node: NodeBase,
    pub children: Vec<ObjRef>,
}

#[derive(Debug, Clone)]
pub enum IndexData {
    Implicit { first: u32, count: u32 },
    BorrowedU8(Rc<[u8]>, Range<usize>),
    BorrowedU16(Rc<[u8]>, Range<usize>),
    BorrowedU32(Rc<[u8]>, Range<usize>),
}

impl IndexData {
    pub fn get(&self, idx: usize) -> u32 {
        match self {
            Self::Implicit { first, .. } => first + idx as u32,
            Self::BorrowedU8(buf, range) => buf[range.start + idx] as u32,
            Self::BorrowedU16(buf, range) => {
                let offset = range.start + idx * 2;
                let bytes = &buf[offset..offset + 2];
                let val = zerocopy::byteorder::little_endian::U16::read_from_bytes(bytes).unwrap();
                val.get() as u32
            }
            Self::BorrowedU32(buf, range) => {
                let offset = range.start + idx * 4;
                let bytes = &buf[offset..offset + 4];
                let val = zerocopy::byteorder::little_endian::U32::read_from_bytes(bytes).unwrap();
                val.get()
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Implicit { count, .. } => *count as usize,
            Self::BorrowedU8(_, range) => range.len(),
            Self::BorrowedU16(_, range) => range.len() / 2,
            Self::BorrowedU32(_, range) => range.len() / 4,
        }
    }
}

#[derive(Debug, Clone)]
pub enum IndexBufferData {
    /// Triangle strips: flattened indices + the length of each strip.
    TriangleStrips {
        strip_lengths: Vec<u32>,
        indices: IndexData,
    },
}

#[derive(Debug, Clone)]
pub struct IndexBuffer {
    pub base: Object3DBase,
    pub data: IndexBufferData,
}

#[derive(Debug, Clone)]
pub enum VertexArrayComponents {
    BorrowedI8(Rc<[u8]>, Range<usize>),
    BorrowedI16(Rc<[u8]>, Range<usize>),
    Owned(Vec<i32>),
}

impl VertexArrayComponents {
    pub fn get(&self, idx: usize) -> i32 {
        match self {
            Self::BorrowedI8(buf, range) => buf[range.start + idx] as i8 as i32,
            Self::BorrowedI16(buf, range) => {
                let offset = range.start + idx * 2;
                let bytes = &buf[offset..offset + 2];
                let val = zerocopy::byteorder::little_endian::I16::read_from_bytes(bytes).unwrap();
                val.get() as i32
            }
            Self::Owned(vec) => vec[idx],
        }
    }
}

#[derive(Debug, Clone)]
pub struct VertexArray {
    pub base: Object3DBase,
    pub component_size: u8,
    pub component_count: u8,
    pub components: VertexArrayComponents,
    pub vertex_count: u16,
}

impl VertexArray {
    pub fn component(&self, idx: usize) -> i32 {
        self.components.get(idx)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TexCoordArray {
    pub array: ObjRef,
    pub bias: [f32; 3],
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct VertexBuffer {
    pub base: Object3DBase,
    pub default_color: [u8; 4],
    pub positions: ObjRef,
    pub position_bias: [f32; 3],
    pub position_scale: f32,
    pub normals: ObjRef,
    pub colors: ObjRef,
    pub tex_coords: Vec<TexCoordArray>,
}

#[derive(Debug, Clone)]
pub struct SubMesh {
    pub index_buffer: ObjRef,
    pub appearance: ObjRef,
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub node: NodeBase,
    pub vertex_buffer: ObjRef,
    pub submeshes: Vec<SubMesh>,
}

#[derive(Debug, Clone)]
pub struct BoneRef {
    pub bone: ObjRef,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub weight: i32,
}

#[derive(Debug, Clone)]
pub struct SkinnedMesh {
    pub mesh: Mesh,
    pub skeleton: ObjRef,
    pub bones: Vec<BoneRef>,
}

#[derive(Debug, Clone)]
pub enum Object {
    Header(Header),
    AnimationController(AnimationController),
    AnimationTrack(AnimationTrack),
    Appearance(Appearance),
    Group(Group),
    Image2D(Image2D),
    TriangleStripArray(IndexBuffer),
    KeyframeSequence(KeyframeSequence),
    Mesh(Mesh),
    SkinnedMesh(SkinnedMesh),
    VertexArray(VertexArray),
    VertexBuffer(VertexBuffer),
    /// Object type recognized but not yet decoded, or unknown: raw payload kept for later.
    Unsupported {
        object_type: u8,
        data: Vec<u8>,
    },
}

pub struct M3GFile {
    /// Index 0 is unused (reserved for the null reference); real objects start at 1.
    pub objects: Vec<Object>,
}

impl M3GFile {
    pub fn get(&self, r: ObjRef) -> Option<&Object> {
        if r.is_null() {
            None
        } else {
            self.objects.get(r.0 as usize)
        }
    }
}

// --- Zerocopy Parsing Structs ---
use zerocopy::byteorder::little_endian::{F32, I32, U16, U32};
use zerocopy::{FromBytes, Immutable, KnownLayout};

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct SectionHeader {
    compression_scheme: u8,
    total_section_length: U32,
    uncompressed_length: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct HeaderHeader {
    version_high: u8,
    version_low: u8,
    has_external_references: u8,
    total_file_size: U32,
    approximate_content_size: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawAnimationController {
    speed: F32,
    weight: F32,
    active_interval_start: U32,
    active_interval_end: U32,
    reference_sequence_time: F32,
    reference_world_time: I32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawAnimationTrack {
    keyframe_sequence: U32,
    animation_controller: U32,
    property_id: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawKeyframeSequence {
    interpolation: u8,
    repeat_mode: u8,
    encoding: u8,
    duration: U32,
    valid_range_first: U32,
    valid_range_last: U32,
    component_count: U32,
    keyframe_count: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawImage2D {
    format: u8,
    is_mutable: u8,
    width: U32,
    height: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawAppearance {
    layer: i8,
    compositing_mode: U32,
    fog: U32,
    polygon_mode: U32,
    material: U32,
    tex_count: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawComponentTransform {
    translation: [F32; 3],
    scale: [F32; 3],
    orientation_angle: F32,
    orientation_axis: [F32; 3],
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawAlignment {
    z_target: u8,
    y_target: u8,
    z_reference: U32,
    y_reference: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawSubMesh {
    index_buffer: U32,
    appearance: U32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawBoneRef {
    bone: U32,
    first_vertex: U32,
    vertex_count: U32,
    weight: I32,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawVertexArray {
    component_size: u8,
    component_count: u8,
    encoding: u8,
    vertex_count: U16,
}

#[derive(FromBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
struct RawTexCoordArray {
    bias: [F32; 3],
    scale: F32,
}

struct Cursor {
    data: Rc<[u8]>,
    pos: usize,
}

impl Cursor {
    fn new(data: Rc<[u8]>) -> Self {
        Cursor { data, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn bytes(&mut self, n: usize) -> &[u8] {
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        s
    }

    fn range(&mut self, n: usize) -> Range<usize> {
        let r = self.pos..self.pos + n;
        self.pos += n;
        r
    }

    fn u8(&mut self) -> u8 {
        let b = self.data[self.pos];
        self.pos += 1;
        b
    }

    fn i8(&mut self) -> i8 {
        self.u8() as i8
    }

    fn bool(&mut self) -> bool {
        self.u8() != 0
    }

    fn read<T: FromBytes + Immutable + KnownLayout>(&mut self) -> T {
        let size = core::mem::size_of::<T>();
        let bytes = &self.data[self.pos..self.pos + size];
        self.pos += size;
        T::read_from_bytes(bytes).unwrap()
    }

    fn u16(&mut self) -> u16 {
        let val: U16 = self.read();
        val.get()
    }

    /// Some fields (e.g. `TriangleStripArray` encoding 2/0x82) are read via a dedicated
    /// `readShortLE` helper in the original loader, distinct from the regular big-endian-ish
    /// `readInt`+byte-swap pattern used elsewhere. Kept as a separate method to document that.
    fn u16_le(&mut self) -> u16 {
        self.u16()
    }

    fn u32(&mut self) -> u32 {
        let val: U32 = self.read();
        val.get()
    }

    fn f32(&mut self) -> f32 {
        let val: F32 = self.read();
        val.get()
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }

    fn obj_ref(&mut self) -> ObjRef {
        ObjRef(self.u32())
    }

    fn string(&mut self) -> String {
        let start = self.pos;
        while self.data[self.pos] != 0 {
            self.pos += 1;
        }
        let s = String::from_utf8_lossy(&self.data[start..self.pos]).into_owned();
        self.pos += 1; // skip NUL
        s
    }
}

fn read_object3d_base(c: &mut Cursor) -> Object3DBase {
    let user_id = c.u32();
    let track_count = c.u32();
    let animation_tracks = (0..track_count).map(|_| c.obj_ref()).collect();
    let param_count = c.u32();
    let mut user_parameters = Vec::with_capacity(param_count as usize);
    for _ in 0..param_count {
        let id = c.u32();
        let len = c.u32() as usize;
        let range = c.range(len);
        let data = ImageBytes::Borrowed(c.data.clone(), range);
        user_parameters.push((id, data));
    }
    Object3DBase {
        user_id,
        animation_tracks,
        user_parameters,
    }
}

/// Order confirmed from `m3g::loadTransformable`/`loadNode` in the game binary: translation
/// then scale then orientation (angle+axis) -- NOT translation/orientation/scale as the public
/// spec text is often paraphrased.
fn read_node_base(c: &mut Cursor) -> NodeBase {
    let base = read_object3d_base(c);

    let has_component_transform = c.bool();
    let (translation, scale, orientation_angle, orientation_axis) = if has_component_transform {
        let raw: RawComponentTransform = c.read();
        (
            [
                raw.translation[0].get(),
                raw.translation[1].get(),
                raw.translation[2].get(),
            ],
            [raw.scale[0].get(), raw.scale[1].get(), raw.scale[2].get()],
            raw.orientation_angle.get(),
            [
                raw.orientation_axis[0].get(),
                raw.orientation_axis[1].get(),
                raw.orientation_axis[2].get(),
            ],
        )
    } else {
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0, [1.0, 0.0, 0.0])
    };
    let has_general_transform = c.bool();
    let general_transform = if has_general_transform {
        let m = c.floats(16);
        Some(m.try_into().unwrap())
    } else {
        None
    };

    let enable_rendering = c.bool();
    let enable_picking = c.bool();
    let alpha_factor = c.u8();
    let scope = c.u32();
    let has_alignment = c.bool();
    let alignment = if has_alignment {
        let raw: RawAlignment = c.read();
        Some(Alignment {
            z_target: raw.z_target,
            y_target: raw.y_target,
            z_reference: ObjRef(raw.z_reference.get()),
            y_reference: ObjRef(raw.y_reference.get()),
        })
    } else {
        None
    };

    NodeBase {
        base,
        enable_rendering,
        enable_picking,
        alpha_factor,
        scope,
        translation,
        orientation_angle,
        orientation_axis,
        scale,
        general_transform,
        alignment,
    }
}

fn read_header(c: &mut Cursor) -> Header {
    let raw: HeaderHeader = c.read();
    let authoring_field = c.string();
    Header {
        version: (raw.version_high, raw.version_low),
        has_external_references: raw.has_external_references != 0,
        total_file_size: raw.total_file_size.get(),
        approximate_content_size: raw.approximate_content_size.get(),
        authoring_field,
    }
}

fn read_animation_controller(c: &mut Cursor) -> AnimationController {
    let base = read_object3d_base(c);
    let raw: RawAnimationController = c.read();
    AnimationController {
        base,
        speed: raw.speed.get(),
        weight: raw.weight.get(),
        active_interval_start: raw.active_interval_start.get(),
        active_interval_end: raw.active_interval_end.get(),
        reference_sequence_time: raw.reference_sequence_time.get(),
        reference_world_time: raw.reference_world_time.get(),
    }
}

fn read_animation_track(c: &mut Cursor) -> AnimationTrack {
    let base = read_object3d_base(c);
    let raw: RawAnimationTrack = c.read();
    AnimationTrack {
        base,
        keyframe_sequence: ObjRef(raw.keyframe_sequence.get()),
        animation_controller: ObjRef(raw.animation_controller.get()),
        property_id: raw.property_id.get(),
    }
}

fn read_keyframe_sequence(c: &mut Cursor) -> KeyframeSequence {
    let base = read_object3d_base(c);
    let raw: RawKeyframeSequence = c.read();
    let interpolation = raw.interpolation;
    let repeat_mode = raw.repeat_mode;
    let encoding = raw.encoding;
    let duration = raw.duration.get();
    let valid_range_first = raw.valid_range_first.get();
    let valid_range_last = raw.valid_range_last.get();
    let component_count = raw.component_count.get() as usize;
    let keyframe_count = raw.keyframe_count.get();

    let data = match encoding {
        0 => {
            let mut frames = Vec::with_capacity(keyframe_count as usize);
            for _ in 0..keyframe_count {
                let time = c.u32();
                frames.push((time, c.floats(component_count)));
            }
            KeyframeData::Float(frames)
        }
        1 | 2 => {
            let bias = c.floats(component_count);
            let scale = c.floats(component_count);
            let mut frames = Vec::with_capacity(keyframe_count as usize);
            for _ in 0..keyframe_count {
                let time = c.u32();
                let comps = (0..component_count)
                    .map(|_| {
                        if encoding == 1 {
                            c.u16() as i32
                        } else {
                            c.i8() as i32
                        }
                    })
                    .collect();
                frames.push((time, comps));
            }
            KeyframeData::Quantized {
                bias,
                scale,
                frames,
            }
        }
        other => panic!("unsupported KeyframeSequence encoding {other}"),
    };

    KeyframeSequence {
        base,
        interpolation,
        repeat_mode,
        duration,
        valid_range_first,
        valid_range_last,
        component_count: component_count as u32,
        keyframe_count,
        data,
    }
}

fn read_image2d(c: &mut Cursor) -> Image2D {
    let base = read_object3d_base(c);
    let raw: RawImage2D = c.read();
    let format = raw.format;
    let is_mutable = raw.is_mutable != 0;
    let width = raw.width.get();
    let height = raw.height.get();
    let (palette, pixels) = if is_mutable {
        (ImageBytes::Empty, ImageBytes::Empty)
    } else {
        let palette_len = c.u32() as usize;
        let palette_range = c.range(palette_len);
        let palette = ImageBytes::Borrowed(c.data.clone(), palette_range);

        let pixels_len = c.u32() as usize;
        let pixels_range = c.range(pixels_len);
        let pixels = ImageBytes::Borrowed(c.data.clone(), pixels_range);
        (palette, pixels)
    };
    Image2D {
        base,
        format,
        is_mutable,
        width,
        height,
        palette,
        pixels,
    }
}

fn read_appearance(c: &mut Cursor) -> Appearance {
    let base = read_object3d_base(c);
    let raw: RawAppearance = c.read();
    let layer = raw.layer;
    let compositing_mode = ObjRef(raw.compositing_mode.get());
    let fog = ObjRef(raw.fog.get());
    let polygon_mode = ObjRef(raw.polygon_mode.get());
    let material = ObjRef(raw.material.get());
    let tex_count = raw.tex_count.get();
    let textures = (0..tex_count).map(|_| c.obj_ref()).collect();
    Appearance {
        base,
        layer,
        compositing_mode,
        fog,
        polygon_mode,
        material,
        textures,
    }
}

fn read_group(c: &mut Cursor) -> Group {
    let node = read_node_base(c);
    let child_count = c.u32();
    let children = (0..child_count).map(|_| c.obj_ref()).collect();
    Group { node, children }
}

fn read_mesh_common(c: &mut Cursor) -> Mesh {
    let node = read_node_base(c);
    let vertex_buffer = c.obj_ref();
    let submesh_count = c.u32();
    let mut submeshes = Vec::with_capacity(submesh_count as usize);
    for _ in 0..submesh_count {
        let raw: RawSubMesh = c.read();
        submeshes.push(SubMesh {
            index_buffer: ObjRef(raw.index_buffer.get()),
            appearance: ObjRef(raw.appearance.get()),
        });
    }
    Mesh {
        node,
        vertex_buffer,
        submeshes,
    }
}

fn read_skinned_mesh(c: &mut Cursor) -> SkinnedMesh {
    let mesh = read_mesh_common(c);
    let skeleton = c.obj_ref();
    let bone_ref_count = c.u32();
    let mut bones = Vec::with_capacity(bone_ref_count as usize);
    for _ in 0..bone_ref_count {
        let raw: RawBoneRef = c.read();
        bones.push(BoneRef {
            bone: ObjRef(raw.bone.get()),
            first_vertex: raw.first_vertex.get(),
            vertex_count: raw.vertex_count.get(),
            weight: raw.weight.get(),
        });
    }
    SkinnedMesh {
        mesh,
        skeleton,
        bones,
    }
}

/// Decodes a `TriangleStripArray` (the only `IndexBuffer` subtype in JSR184).
///
/// Field layout reverse-engineered from `m3g::Loader::load` in the game binary (case 0xb),
/// since it does not match the assumed-from-spec layout: `encoding` selects one of two
/// unrelated shapes, and a strip-length array *always* follows, regardless of encoding.
///   - 0/1/2: a single implicit `firstIndex` (as i32 / u8 / u16-LE respectively); strips are
///     `firstIndex, firstIndex+1, ...` contiguous vertex indices.
///   - 0x80/0x81/0x82: an explicit index count (u32) followed by that many indices, stored
///     as i32 / u8 / u16-LE respectively.
///   - anything else: no indices/firstIndex field at all.
/// Then always: stripLengthCount (u32) + that many u32 strip lengths.
fn read_triangle_strip_array(c: &mut Cursor) -> IndexBuffer {
    let base = read_object3d_base(c);
    let encoding = c.u8();

    let mut explicit_indices: Option<IndexData> = None;
    let mut first_index: Option<u32> = None;
    match encoding {
        0 => first_index = Some(c.u32()),
        1 => first_index = Some(c.u8() as u32),
        2 => first_index = Some(c.u16_le() as u32),
        0x80 => {
            let n = c.u32() as usize;
            let range = c.range(n * 4);
            explicit_indices = Some(IndexData::BorrowedU32(c.data.clone(), range));
        }
        0x81 => {
            let n = c.u32() as usize;
            let range = c.range(n);
            explicit_indices = Some(IndexData::BorrowedU8(c.data.clone(), range));
        }
        0x82 => {
            let n = c.u32() as usize;
            let range = c.range(n * 2);
            explicit_indices = Some(IndexData::BorrowedU16(c.data.clone(), range));
        }
        _ => {}
    }

    let strip_length_count = c.u32() as usize;
    let strip_lengths: Vec<u32> = (0..strip_length_count).map(|_| c.u32()).collect();

    let indices = if let Some(indices) = explicit_indices {
        indices
    } else {
        let first = first_index.unwrap_or(0);
        let total: u32 = strip_lengths.iter().sum();
        IndexData::Implicit {
            first,
            count: total,
        }
    };

    IndexBuffer {
        base,
        data: IndexBufferData::TriangleStrips {
            strip_lengths,
            indices,
        },
    }
}

fn read_vertex_array(c: &mut Cursor) -> VertexArray {
    let base = read_object3d_base(c);
    let raw: RawVertexArray = c.read();
    let component_size = raw.component_size;
    let component_count = raw.component_count;
    let encoding = raw.encoding;
    let vertex_count = raw.vertex_count.get();

    let n = vertex_count as usize * component_count as usize;
    let components = match component_size {
        1 => {
            let range = c.range(n);
            if encoding == 0 {
                VertexArrayComponents::BorrowedI8(c.data.clone(), range)
            } else if encoding == 1 {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(c.data[range.start + i] as i8 as i32);
                }
                for v in 1..vertex_count as usize {
                    for k in 0..component_count as usize {
                        let idx = v * component_count as usize + k;
                        out[idx] = out[idx - component_count as usize] + out[idx];
                    }
                }
                VertexArrayComponents::Owned(out)
            } else {
                panic!("unsupported VertexArray encoding {encoding}");
            }
        }
        2 => {
            let range = c.range(n * 2);
            if encoding == 0 {
                VertexArrayComponents::BorrowedI16(c.data.clone(), range)
            } else if encoding == 1 {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let offset = range.start + i * 2;
                    let bytes = &c.data[offset..offset + 2];
                    let val =
                        zerocopy::byteorder::little_endian::I16::read_from_bytes(bytes).unwrap();
                    out.push(val.get() as i32);
                }
                for v in 1..vertex_count as usize {
                    for k in 0..component_count as usize {
                        let idx = v * component_count as usize + k;
                        out[idx] = out[idx - component_count as usize] + out[idx];
                    }
                }
                VertexArrayComponents::Owned(out)
            } else {
                panic!("unsupported VertexArray encoding {encoding}");
            }
        }
        other => panic!("unsupported VertexArray componentSize {other}"),
    };

    VertexArray {
        base,
        component_size,
        component_count,
        components,
        vertex_count,
    }
}

fn read_tex_coord_array(c: &mut Cursor) -> TexCoordArray {
    let array = c.obj_ref();
    if array.is_null() {
        return TexCoordArray::default();
    }
    let raw: RawTexCoordArray = c.read();
    TexCoordArray {
        array,
        bias: [raw.bias[0].get(), raw.bias[1].get(), raw.bias[2].get()],
        scale: raw.scale.get(),
    }
}

fn read_vertex_buffer(c: &mut Cursor) -> VertexBuffer {
    let base = read_object3d_base(c);
    let default_color = {
        let b = c.bytes(4);
        [b[0], b[1], b[2], b[3]]
    };
    let positions = c.obj_ref();
    let (position_bias, position_scale) = if positions.is_null() {
        ([0.0; 3], 1.0)
    } else {
        let raw: RawTexCoordArray = c.read();
        (
            [raw.bias[0].get(), raw.bias[1].get(), raw.bias[2].get()],
            raw.scale.get(),
        )
    };
    let normals = c.obj_ref();
    let colors = c.obj_ref();
    let tex_coord_count = c.u32();
    let tex_coords = (0..tex_coord_count)
        .map(|_| read_tex_coord_array(c))
        .collect();
    VertexBuffer {
        base,
        default_color,
        positions,
        position_bias,
        position_scale,
        normals,
        colors,
        tex_coords,
    }
}

fn read_object(object_type: u8, data: &[u8]) -> Object {
    let mut c = Cursor::new(Rc::from(data));
    match object_type {
        0 => Object::Header(read_header(&mut c)),
        1 => Object::AnimationController(read_animation_controller(&mut c)),
        2 => Object::AnimationTrack(read_animation_track(&mut c)),
        3 => Object::Appearance(read_appearance(&mut c)),
        9 => Object::Group(read_group(&mut c)),
        10 => Object::Image2D(read_image2d(&mut c)),
        11 => Object::TriangleStripArray(read_triangle_strip_array(&mut c)),
        14 => Object::Mesh(read_mesh_common(&mut c)),
        16 => Object::SkinnedMesh(read_skinned_mesh(&mut c)),
        19 => Object::KeyframeSequence(read_keyframe_sequence(&mut c)),
        20 => Object::VertexArray(read_vertex_array(&mut c)),
        21 => Object::VertexBuffer(read_vertex_buffer(&mut c)),
        _ => Object::Unsupported {
            object_type,
            data: data.to_vec(),
        },
    }
}

fn decompress_section_body(compression_scheme: u8, body: &[u8], uncompressed_len: u32) -> Rc<[u8]> {
    match compression_scheme {
        0 => body.into(),
        1 => {
            let mut out = vec![0u8; uncompressed_len as usize];
            let (_, rc) =
                zlib_rs::decompress_slice(&mut out, body, zlib_rs::InflateConfig::default());
            assert_eq!(rc, zlib_rs::ReturnCode::Ok, "zlib decompression failed");
            out.into()
        }
        other => panic!("unsupported section compression scheme {other}"),
    }
}

pub fn parse(data: &[u8]) -> M3GFile {
    assert!(data.len() >= 12 && data[..12] == MAGIC, "not an M3G file");

    let mut objects: Vec<Object> = vec![Object::Unsupported {
        object_type: 0,
        data: Vec::new(),
    }]; // index 0 = null placeholder

    let mut pos = 12usize;
    while pos < data.len() {
        let (header, _) =
            SectionHeader::ref_from_prefix(&data[pos..]).expect("truncated section header");
        let compression_scheme = header.compression_scheme;
        let total_section_length = header.total_section_length.get() as usize;
        let uncompressed_length = header.uncompressed_length.get();

        let body_start = pos + 9;
        let body_len = total_section_length - 9 - 4; // minus header and trailing checksum
        let body = &data[body_start..body_start + body_len];
        let objects_data = decompress_section_body(compression_scheme, body, uncompressed_length);

        let mut c = Cursor::new(objects_data);
        while !c.eof() {
            let object_type = c.u8();
            let len = c.u32() as usize;
            let range = c.range(len);
            let obj_data = &c.data[range];
            objects.push(read_object(object_type, obj_data));
        }

        pos = body_start + body_len + 4; // skip checksum
    }

    M3GFile { objects }
}
