from sennet.core.mmap_arrays import read_mmap_array, create_mmap_array
from pathlib import Path
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--seed-threshold", type=float, required=True, default=0.5)
    parser.add_argument("--seed-dir-name", type=str, required=False, default="seed")
    parser.add_argument("--out-dir-name", type=str, required=False, default="out")
    parser.add_argument("--skip-seed", action="store_true", required=False,  default=False)
    args, _ = parser.parse_known_args()
    path = Path(args.path)
    skip_seed = args.skip_seed
    seed_threshold = args.seed_threshold
    seed_dir_name = args.seed_dir_name
    out_dir_name = args.out_dir_name

    sub_dirs = list(path.glob("*"))
    for sub_dir in tqdm(sub_dirs):
        mean_prob_dir = sub_dir / "chunk_00" / "mean_prob"
        mean_prob = read_mmap_array(mean_prob_dir, mode="r").data

        out_mmap = create_mmap_array(sub_dir / "chunk_00" / out_dir_name, shape=list(mean_prob.shape), dtype=bool)
        out_mmap.data[:] = False
        out_mmap.data.flush()

        if not skip_seed:
            seed_mmap = create_mmap_array(sub_dir / "chunk_00" / seed_dir_name, shape=list(mean_prob.shape), dtype=bool)
            seed_mmap.data[:] = mean_prob.data[:] > seed_threshold
            seed_mmap.data.flush()
    print("done")


if __name__ == "__main__":
    main()
