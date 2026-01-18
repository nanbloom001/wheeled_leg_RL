import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="Custom URDF Converter.")
parser.add_argument("input_file", type=str, help="Input URDF file.")
parser.add_argument("output_file", type=str, help="Output USD file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

def main():
    # 配置转换器
    # 必须提供所有必需字段
    drive_cfg = UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0)
    )

    cfg = UrdfConverterCfg(
        asset_path=args_cli.input_file,
        usd_dir=args_cli.output_file.rsplit("/", 1)[0],
        usd_file_name=args_cli.output_file.rsplit("/", 1)[1],
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=drive_cfg,
        collider_type="convex_decomposition"  # 显式指定
    )

    print(f"[INFO] Converting {args_cli.input_file} -> {args_cli.output_file}")
    
    # 实例化转换器，这会自动触发转换
    converter = UrdfConverter(cfg)
    
    print("[INFO] Conversion complete.")

if __name__ == "__main__":
    main()
    simulation_app.close()