import csv
from pathlib import Path

import mujoco


def main():
    root = Path('/home/user/IsaacLab')
    xml_path = root / 'WAVEGO_mujoco' / 'scene.xml'
    out_csv = root / 'tmp' / 'mujoco_compiled_body_mass_inertia.csv'
    out_txt = root / 'tmp' / 'mujoco_compiled_summary.txt'
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    rows = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f'body_{body_id}'
        mass = float(model.body_mass[body_id])
        inertia = [float(x) for x in model.body_inertia[body_id]]
        ipos = [float(x) for x in model.body_ipos[body_id]]
        iquat = [float(x) for x in model.body_iquat[body_id]]
        parent = int(model.body_parentid[body_id])
        rows.append([body_id, name, parent, mass, *inertia, *ipos, *iquat])

    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'body_id', 'body_name', 'parent_id', 'mass',
            'inertia_xx', 'inertia_yy', 'inertia_zz',
            'ipos_x', 'ipos_y', 'ipos_z',
            'iquat_w', 'iquat_x', 'iquat_y', 'iquat_z'
        ])
        writer.writerows(rows)

    total_mass = sum(float(x) for x in model.body_mass)
    with out_txt.open('w') as f:
        f.write(f'xml={xml_path}\n')
        f.write(f'nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, nq={model.nq}, nv={model.nv}\n')
        f.write(f'timestep={model.opt.timestep}\n')
        f.write(f'gravity={list(float(x) for x in model.opt.gravity)}\n')
        f.write(f'total_body_mass={total_mass:.9f}\n')

    print(f'Wrote: {out_csv}')
    print(f'Wrote: {out_txt}')
    print(f'total_body_mass={total_mass:.9f}')


if __name__ == '__main__':
    main()
