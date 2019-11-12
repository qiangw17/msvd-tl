# msvd-tl
More informative semantic cues might be exploited via a multirounds chatting or discussing about the video among multiple speakers. So multi-speakers video dialogs are more applicable in real life. We introduce a novel task Multi-Speaker Video Dialog with frame-level Temporal Localization (MSVD-TL). It targets to predict the following response and localize the relevant video sub-segment in frame level, simultaneously. And we focus on the characteristic of the video dialog generation process and exploit the relation among the video fragment, the chat history, and the following response to refine their representations.

- The flowchart of the approach.

![image](https://github.com/qiangw17/msvd-tl/raw/master/images/framework.jpg)



## Dataset

[the raw videos at 3fps](https://drive.google.com/drive/folders/11VE_uDByvF5AkVD8QEoU5VlDlRGmsDv1)

[train_partition_fc_50s.txt](https://drive.google.com/open?id=1gwizfZNP0C0rvvsUb063zqc7Mr3Lq_K-)

[val_partition_fc_50s.txt](https://drive.google.com/open?id=172qtP7MrZNg6ZmWyEFeFzfCBydjYMACn)

[test_partition_fc_50s.txt](https://drive.google.com/open?id=1QRcg688XksyG7Z51YnRfHv9SRi8s0yAy)

[Twitch-FIFA-Dataset](https://drive.google.com/open?id=1ZCovUXqLgPBZOmXNEUC9YSWJUrNl2J3v)



## Run Code

```
python retrievalTrainAndTest.py --word_level --video_context --chat_context --model_name 50s --use_glove
```



## Paper

Qiang Wang, Pin Jiang, Yahong Han, Zhou Zhao. ["Multi-Speaker Video Dialog with Frame-Level Temporal Localization."] AAAI, 2020. 
```
@inproceedings{Wang2020,
  author    = {Qiang Wang, Pin Jiang, Yahong Han, Zhou Zhao},
  title     = {Multi-Speaker Video Dialog with Frame-Level Temporal Localization},
  booktitle = {AAAI},
  year      = {2020},
}
```

