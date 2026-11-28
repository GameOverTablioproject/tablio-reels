"""Needle-over-balloon scene (photoreal, Cycles). Usage:
python3 scene.py OUTDIR FRAME_START FRAME_END [RES_PCT] [SAMPLES]
Frames are 1-based; contact would happen at frame CONTACT (never shown)."""
import bpy, math, sys, os
from mathutils import Vector

out, f0, f1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pct = int(sys.argv[4]) if len(sys.argv) > 4 else 100
spp = int(sys.argv[5]) if len(sys.argv) > 5 else 32

FPS = 30
N = 210                      # frames shown (7.0 s); counter = CONTACT - frame
CONTACT = N + 1              # contact would be the frame after the last one
R = 0.12                     # balloon radius (m)
TOP = 2 * R * 1.08           # balloon top height (z) after stretch
START_GAP = 0.36             # needle tip height above balloon top at frame 1
NEEDLE_L = 0.11

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, N

# ---------------------------------------------------------------- materials
def mat(name, **kw):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    for k, v in kw.items(): b.inputs[k].default_value = v
    return m

latex = mat('latex', **{'Base Color': (0.30, 0.004, 0.007, 1), 'Roughness': 0.22,
                        'Coat Weight': 0.7, 'Coat Roughness': 0.03, 'Specular IOR Level': 0.5,
                        'Subsurface Weight': 0.0, 'Subsurface Radius': (1.0, 0.03, 0.03), 'Subsurface Scale': 0.006})
steel = mat('steel', **{'Base Color': (0.82, 0.82, 0.84, 1), 'Metallic': 1.0, 'Roughness': 0.13, 'Anisotropic': 0.6})
table = mat('table', **{'Base Color': (0.006, 0.006, 0.007, 1), 'Roughness': 0.7, 'Specular IOR Level': 0.12})
# subtle large-scale roughness variation on the table so reflections read as a real surface
nt = table.node_tree
tex = nt.nodes.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value = 3.0
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['To Min'].default_value = 0.4; mr.inputs['To Max'].default_value = 0.65
nt.links.new(tex.outputs['Fac'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], nt.nodes['Principled BSDF'].inputs['Roughness'])

# ---------------------------------------------------------------- balloon (sphere, teardrop bottom, knot)
bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=R, location=(0, 0, 0))
bal = bpy.context.object; bal.name = 'balloon'
me = bal.data
for v in me.vertices:
    z = v.co.z / R                               # -1..1
    if z < 0:                                    # taper the bottom into a teardrop
        k = 1 - 0.38 * (-z) ** 2.2
        v.co.x *= k; v.co.y *= k; v.co.z *= 1.18
    else:
        v.co.z *= 1.03
for p in me.polygons: p.use_smooth = True
bal.data.materials.append(latex)
zmin = min(v.co.z for v in me.vertices)
# knot
bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.006, radius2=0.017, depth=0.022, location=(0, 0, zmin - 0.006))
knot = bpy.context.object; knot.data.materials.append(latex)
for p in knot.data.polygons: p.use_smooth = True
bpy.ops.mesh.primitive_torus_add(major_radius=0.008, minor_radius=0.0045, location=(0, 0, zmin - 0.019))
ring = bpy.context.object; ring.data.materials.append(latex)
for p in ring.data.polygons: p.use_smooth = True
# lift everything so the knot rests on the table, tilt slightly (resting balloon leans)
lift = -(zmin - 0.0235)
for o in (bal, knot, ring):
    o.location.z += lift
bpy.ops.object.empty_add(location=(0, 0, 0)); piv = bpy.context.object
for o in (bal, knot, ring): o.parent = piv
piv.rotation_euler = (math.radians(9), math.radians(-4), 0)
bpy.context.view_layer.update()
top_z = max((bal.matrix_world @ v.co).z for v in me.vertices)
# point on the balloon top that the needle aims at
top_pt = max((bal.matrix_world @ v.co for v in me.vertices), key=lambda c: c.z)

# ---------------------------------------------------------------- needle (steel, eye, sharp tip)
def needle():
    rad = 0.0012
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=rad, depth=NEEDLE_L - 0.012, location=(0, 0, 0.012 + (NEEDLE_L - 0.012) / 2))
    shaft = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=rad, radius2=0.00002, depth=0.012, location=(0, 0, 0.006))
    tip = bpy.context.object; tip.rotation_euler = (math.pi, 0, 0)
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=rad * 1.15, depth=0.007, location=(0, 0, NEEDLE_L - 0.0035))
    head = bpy.context.object
    for o in (shaft, tip, head):
        o.data.materials.append(steel)
        for p in o.data.polygons: p.use_smooth = True
    # eye: boolean slot through the head
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, NEEDLE_L - 0.0042))
    eye = bpy.context.object; eye.scale = (0.00034, 0.01, 0.0026)
    bm = head.modifiers.new('eye', 'BOOLEAN'); bm.object = eye; bm.operation = 'DIFFERENCE'
    eye.hide_render = True; eye.hide_viewport = True
    bpy.ops.object.empty_add(location=(0, 0, 0)); root = bpy.context.object; root.name = 'needle'
    for o in (shaft, tip, head, eye): o.parent = root
    for o in (shaft, tip, head):
        o.modifiers.new('bev', 'WEIGHTED_NORMAL')
    return root
nd = needle()

def tip_height(f):
    """gap between needle tip and balloon top at frame f; linear in frames so the
    on-screen frame counter is literally the frames left until contact."""
    return START_GAP * (CONTACT - f) / (CONTACT - 1)

for f in range(1, N + 1):
    g = tip_height(f)
    s = (f - 1) / (N - 1)
    nd.location = (top_pt.x + 0.0006 * math.sin(f * 0.09), top_pt.y, top_pt.z + g)
    # tiny slow tumble that settles as it nears the surface (slow-motion fall)
    w = 1 - s
    nd.rotation_euler = (math.radians(2.2 * w * math.sin(f * 0.035)), math.radians(1.6 * w * math.cos(f * 0.028)), f * 0.004)
    nd.keyframe_insert('location', frame=f); nd.keyframe_insert('rotation_euler', frame=f)

# ---------------------------------------------------------------- table + backdrop
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0)); tbl = bpy.context.object; tbl.data.materials.append(table)
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 2.4, 2), rotation=(math.radians(90), 0, 0))
wall = bpy.context.object
wm = mat('wall', **{'Base Color': (0.004, 0.004, 0.005, 1), 'Roughness': 0.9}); wall.data.materials.append(wm)

world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.004, 0.005, 1)

# ---------------------------------------------------------------- lights (Fern-ish: one warm key, cool rim, faint top)
def area(name, loc, size, power, color, target=(0, 0, 0.2)):
    bpy.ops.object.light_add(type='AREA', location=loc)
    L = bpy.context.object; L.name = name; L.data.size = size; L.data.energy = power * float(os.environ.get('LS', '0.12')); L.data.color = color
    d = Vector(target) - Vector(loc); L.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return L
k = area('key', (-0.8, -0.6, 1.0), 0.7, 75, (1.0, 0.84, 0.68)); k.data.spread = math.radians(55)
area('rim', (0.5, 0.8, 0.95), 0.4, 160, (0.6, 0.76, 1.0), target=(0, 0, 0.3))
t2 = area('top', (0.0, 0.1, 1.4), 0.3, 0, (1.0, 0.95, 0.9), target=(0, 0, 0.5)); t2.data.spread = math.radians(35)
st = area('strip', (-0.32, -0.36, 0.52), 0.022, 110, (1.0, 0.97, 0.93), target=(0, 0, 0.5)); st.data.shape = 'RECTANGLE'; st.data.size_y = 0.9
st2 = area('strip2', (0.36, -0.30, 0.55), 0.018, 55, (0.75, 0.85, 1.0), target=(0, 0, 0.5)); st2.data.shape = 'RECTANGLE'; st2.data.size_y = 0.9
wg = area('wallglow', (0.0, 1.5, 0.2), 0.8, 90, (1.0, 0.75, 0.55), target=(0.1, 2.4, 0.8))

# ---------------------------------------------------------------- camera: slow push-in, then accelerating macro zoom on the gap
bpy.ops.object.camera_add(); cam = bpy.context.object; sc.camera = cam
cam.data.sensor_width = 36; cam.data.dof.use_dof = True; cam.data.dof.aperture_fstop = 2.8
bpy.ops.object.empty_add(location=(0, 0, 0)); aim = bpy.context.object
cam.constraints.new('TRACK_TO').target = aim
cam.constraints['Track To'].track_axis = 'TRACK_NEGATIVE_Z'; cam.constraints['Track To'].up_axis = 'UP_Y'
bpy.ops.object.empty_add(location=(0, 0, 0)); foc = bpy.context.object; cam.data.dof.focus_object = foc

def ease_in(x, p): return x ** p
for f in range(1, N + 1):
    s = (f - 1) / (N - 1)
    g = tip_height(f)
    tip = Vector((top_pt.x, top_pt.y, top_pt.z + g))
    mid = Vector((top_pt.x, top_pt.y, top_pt.z + g * 0.5))
    # aim: starts between balloon centre and needle, ends on the contact point
    a = ease_in(s, 1.6)
    centre = Vector((0, 0, top_z * 0.62 + START_GAP * 0.35))
    aim.location = centre.lerp(mid, a)
    # camera distance shrinks: gentle dolly, then fast in the last ~25%
    dist = 1.55 - 0.55 * s - 0.78 * ease_in(s, 5.0)
    height = 0.42 + 0.12 * s - 0.0 * s
    ang = math.radians(-18 + 10 * s)                      # slow orbit
    cam.location = (aim.location.x + dist * math.sin(ang), aim.location.y - dist * math.cos(ang), aim.location.z + (height - aim.location.z) * (1 - a) + 0.02 * a)
    cam.data.lens = 50 + 40 * s + 110 * ease_in(s, 4.0)     # zoom: 50 mm -> 200 mm
    # micro shake grows with tension
    sh = 0.0004 * ease_in(s, 3)
    cam.location.x += sh * math.sin(f * 1.7); cam.location.z += sh * math.cos(f * 2.3)
    foc.location = tip - Vector((0, 0, g * 0.45 * a))
    cam.data.dof.aperture_fstop = 2.8 + 5.2 * ease_in(s, 2)  # keep both tip and latex sharp-ish at macro
    for o, p in ((aim, 'location'), (cam, 'location'), (foc, 'location')): o.keyframe_insert(p, frame=f)
    cam.data.keyframe_insert('lens', frame=f); cam.data.dof.keyframe_insert('aperture_fstop', frame=f)

# ---------------------------------------------------------------- render settings
r = sc.render
r.engine = 'CYCLES'; sc.cycles.device = 'CPU'
sc.cycles.samples = spp; sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.03
sc.cycles.use_denoising = True; sc.cycles.denoiser = 'OPENIMAGEDENOISE'
sc.cycles.max_bounces = 6; sc.cycles.glossy_bounces = 4; sc.cycles.transmission_bounces = 2; sc.cycles.diffuse_bounces = 2
sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
r.resolution_x, r.resolution_y, r.resolution_percentage = 1080, 1920, pct
r.image_settings.file_format = 'PNG'; r.image_settings.color_mode = 'RGB'
sc.view_settings.view_transform = 'Khronos PBR Neutral'
sc.view_settings.exposure = float(os.environ.get('EXP', '0'))
r.threads_mode = 'AUTO'
os.makedirs(out, exist_ok=True)
for f in range(f0, f1 + 1):
    sc.frame_set(f)
    r.filepath = os.path.join(out, f'{f:04d}.png')
    bpy.ops.render.render(write_still=True)
    print('frame', f, flush=True)
