import sys
from pathlib import Path


current_dir = Path(__file__).parent
project_root = current_dir.parent

current_dir_str = str(current_dir)
if current_dir_str in sys.path:
    sys.path.remove(current_dir_str)

project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

from finetune.models.utils import get_model_cls
from finetune.schemas import Args


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
