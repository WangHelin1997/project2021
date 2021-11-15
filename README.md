# project2021
PKU team for 2021 project 'Guangchangwu detection'.

## Note
To test or train, change the paths in the .py files to your own paths.

## Activate environment
```
cd /home/pkusz/home/PKU_team/pku_code/
su
conda activate pku
```

## To Test

Test single audio file (Now support 5-second audio)
```
sh test_file.sh
```

Test audio files (Using file path)
```
sh test_files.sh
```

### To Train

Make dataset
```
sh make_dataset.sh
```

Train model
```
sh train.sh
```


## Authors:
Helin Wang, Zhongjie Ye, Dongchao Yang
