# Adding a New Dataset for Testing

Download your new dataset

## File/Folders to add

**Example:** *[DATASET_NAME] = **toronto3d** (for the Toronto3D dataset)*

- #### root/config/[DATASET_NAME]_COSeg_fs.yaml
    (File for configuring the dataset and model)

- #### root/datasets/[DATASET_NAME]_classnames.txt
    (Defines the classnames for the dataset)

- #### root/preprocess/collect_[DATASET_NAME]_data.py
    (Script for preprocessing the dataset into .npy files)

- #### root/util/[DATASET_NAME]_fs.py
    (Defines the framework for handling the dataset)

## Files/Folders to modify

- #### root/main_fs.py

Add the following block of code after line 305:

```
elif args.data_name == "[DATASET_NAME]":
        val_data = [DATASET_NAME]_FS_TEST(
            split=args.eval_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=val_transform,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
            num_episode_per_comb=args.num_episode_per_comb,
        )
        valid_calsses = list(val_data.classes)
```

## Next Steps

- Follow the steps under "Dataset Preperation" in "README.md"

- After running the preprocessing steps add the folder/file **meta/[DATASET_NAME]_classnames.txt** to **[PATH_to_processed_data]/scenes**