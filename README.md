# msvd-tl
More informative semantic cues might be exploited via a multirounds chatting or discussing about the video among multiple speakers. So multi-speakers video dialogs are more applicable in real life. We introduce a novel task Multi-Speaker Video Dialog with frame-level Temporal Localization (MSVD-TL). It targets to predict the following response and localize the relevant video sub-segment in frame level, simultaneously. And we focus on the characteristic of the video dialog generation process and exploit the relation among the video fragment, the chat history, and the following response to refine their representations.

- The flowchart of our approach for MSVD-TL.

![image](https://github.com/qiangw17/msvd-tl/raw/master/images/framework.jpg)



## Dataset

You can download the raw videos at 3fps from [here](https://drive.google.com/drive/folders/11VE_uDByvF5AkVD8QEoU5VlDlRGmsDv1).

You can download the file with frame indexs for all training samples from [train_partition_fc_50s.txt](https://drive.google.com/open?id=1gwizfZNP0C0rvvsUb063zqc7Mr3Lq_K-).

You can download the file with frame indexs for all validation samples from [val_partition_fc_50s.txt](https://drive.google.com/open?id=172qtP7MrZNg6ZmWyEFeFzfCBydjYMACn).

You can download the file with frame indexs for all tesing samples from [test_partition_fc_50s.txt](https://drive.google.com/open?id=1QRcg688XksyG7Z51YnRfHv9SRi8s0yAy).

You can download the full processed dataset from [Twitch-FIFA-Dataset](https://drive.google.com/open?id=1ZCovUXqLgPBZOmXNEUC9YSWJUrNl2J3v).



## Run Code

```
python retrievalTrainAndTest.py --word_level --video_context --chat_context --model_name 50s --use_glove
```



## Paper

Qiang Wang, Pin Jiang, Zhiyi Guo, Yahong Han, Zhou Zhao. ["Multi-Speaker Video Dialog with Frame-Level Temporal Localization."] AAAI, 2020. 
```
@inproceedings{Wang2020,
  author    = {Qiang Wang, Pin Jiang, Zhiyi Guo, Yahong Han, Zhou Zhao},
  title     = {Multi-Speaker Video Dialog with Frame-Level Temporal Localization},
  booktitle = {AAAI},
  year      = {2020},
}
```

