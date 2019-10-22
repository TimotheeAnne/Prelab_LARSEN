path = "./exp/log/Saved/model_size/"
file = '/logs.mat'

test= ['2019-10-11--15:27:0067542']
small = ['2019-10-11--10:16:1556039']

model0 = ['2019-10-11--10:28:567824']
model1 = ['2019-10-11--10:28:5610303']
model2 = ['2019-10-11--10:28:5689837']
model3 = ['2019-10-11--10:28:5652030']
model4 = ['2019-10-11--10:28:568382']
model5 = ['2019-10-11--10:28:5669595']
model6 = ['2019-10-11--10:28:56856']
model7 = ['2019-10-11--10:28:5681889']
model8 = ['2019-10-11--10:28:5690694']
model9 = ['2019-10-11--10:28:5630246']
model10 = ['2019-10-11--10:29:1717079']

subdirs = [
        [path + x + '/logs.mat' for x in model0],
        [path + x + '/logs.mat' for x in model1],
        [path + x + '/logs.mat' for x in model2],
        [path + x + '/logs.mat' for x in model3],
        [path + x + '/logs.mat' for x in model4],
        [path + x + '/logs.mat' for x in model5],
        [path + x + '/logs.mat' for x in model6],
        [path + x + '/logs.mat' for x in model7],
        [path + x + '/logs.mat' for x in model8],
        [path + x + '/logs.mat' for x in model9],
        [path + x + '/logs.mat' for x in model10],
]

tan50 = [
['./exp/log/Saved/21_10_01/model_tan0_50/model_tan0_50_replicate_0/2019-10-11--23:39:1946788/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan1_50/model_tan1_50_replicate_0/2019-10-11--23:44:2429465/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan2_50/model_tan2_50_replicate_0/2019-10-11--23:44:4620581/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan3_50/model_tan3_50_replicate_0/2019-10-11--23:47:1783418/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan4_50/model_tan4_50_replicate_0/2019-10-11--23:51:4724421/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan5_50/model_tan5_50_replicate_0/2019-10-11--23:59:1669335/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan6_50/model_tan6_50_replicate_0/2019-10-12--00:04:4055109/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan7_50/model_tan7_50_replicate_0/2019-10-12--00:05:2018887/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan8_50/model_tan8_50_replicate_0/2019-10-12--00:07:5124355/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan9_50/model_tan9_50_replicate_0/2019-10-12--00:12:3674174/logs.mat', ],
]

tan100 = [
['./exp/log/Saved/21_10_01/model_tan0_100/model_tan0_100_replicate_0/2019-10-11--22:42:256418/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan1_100/model_tan1_100_replicate_0/2019-10-11--22:42:4592458/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan2_100/model_tan2_100_replicate_0/2019-10-11--22:45:4297135/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan3_100/model_tan3_100_replicate_0/2019-10-11--22:51:2829289/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan4_100/model_tan4_100_replicate_0/2019-10-11--22:51:5214223/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan5_100/model_tan5_100_replicate_0/2019-10-11--22:58:2856624/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan6_100/model_tan6_100_replicate_0/2019-10-11--23:03:0648649/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan7_100/model_tan7_100_replicate_0/2019-10-11--23:03:0460702/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan8_100/model_tan8_100_replicate_0/2019-10-11--23:03:5285672/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan9_100/model_tan9_100_replicate_0/2019-10-11--23:06:586306/logs.mat', ],
]

tan200 = [
['./exp/log/Saved/21_10_01/model_tan0_200/model_tan0_200_replicate_0/2019-10-11--23:06:5812105/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan1_200/model_tan1_200_replicate_0/2019-10-11--23:08:2437771/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan2_200/model_tan2_200_replicate_0/2019-10-11--23:11:5680231/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan3_200/model_tan3_200_replicate_0/2019-10-11--23:18:0462260/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan4_200/model_tan4_200_replicate_0/2019-10-11--23:22:1422418/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan5_200/model_tan5_200_replicate_0/2019-10-11--23:22:36111/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan6_200/model_tan6_200_replicate_0/2019-10-11--23:24:3611206/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan7_200/model_tan7_200_replicate_0/2019-10-11--23:29:2749976/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan8_200/model_tan8_200_replicate_0/2019-10-11--23:33:0424359/logs.mat', ],
['./exp/log/Saved/21_10_01/model_tan9_200/model_tan9_200_replicate_0/2019-10-11--23:33:1954227/logs.mat', ],
]

relu200 = [
['./exp/log/Saved/21_10_01/model0_200/model0_200_replicate_0/2019-10-11--21:41:4957657/logs.mat', ],
['./exp/log/Saved/21_10_01/model1_200/model1_200_replicate_0/2019-10-11--21:44:4389236/logs.mat', ],
['./exp/log/Saved/21_10_01/model2_200/model2_200_replicate_0/2019-10-11--21:45:0841961/logs.mat', ],
['./exp/log/Saved/21_10_01/model3_200/model3_200_replicate_0/2019-10-11--21:45:0837668/logs.mat', ],
['./exp/log/Saved/21_10_01/model4_200/model4_200_replicate_0/2019-10-11--21:45:0854999/logs.mat', ],
['./exp/log/Saved/21_10_01/model5_200/model5_200_replicate_0/2019-10-11--21:45:087757/logs.mat', ],
['./exp/log/Saved/21_10_01/model6_200/model6_200_replicate_0/2019-10-11--21:46:437377/logs.mat', ],
['./exp/log/Saved/21_10_01/model7_200/model7_200_replicate_0/2019-10-11--21:46:433170/logs.mat', ],
['./exp/log/Saved/21_10_01/model8_200/model8_200_replicate_0/2019-10-11--21:46:4375052/logs.mat', ],
['./exp/log/Saved/21_10_01/model9_200/model9_200_replicate_0/2019-10-11--21:46:4360284/logs.mat', ],
]

relu100 = [
['./exp/log/Saved/21_10_01/model0_100/model0_100_replicate_0/2019-10-11--19:31:5075353/logs.mat', ],
['./exp/log/Saved/21_10_01/model1_100/model1_100_replicate_0/2019-10-11--19:31:5062337/logs.mat', ],
['./exp/log/Saved/21_10_01/model2_100/model2_100_replicate_0/2019-10-11--19:31:5047909/logs.mat', ],
['./exp/log/Saved/21_10_01/model3_100/model3_100_replicate_0/2019-10-11--19:31:5013531/logs.mat', ],
['./exp/log/Saved/21_10_01/model4_100/model4_100_replicate_0/2019-10-11--20:59:5122734/logs.mat', ],
['./exp/log/Saved/21_10_01/model5_100/model5_100_replicate_0/2019-10-11--20:59:5116199/logs.mat', ],
['./exp/log/Saved/21_10_01/model6_100/model6_100_replicate_0/2019-10-11--20:59:5161954/logs.mat', ],
['./exp/log/Saved/21_10_01/model7_100/model7_100_replicate_0/2019-10-11--20:59:512165/logs.mat', ],
['./exp/log/Saved/21_10_01/model8_100/model8_100_replicate_0/2019-10-11--21:41:0271909/logs.mat', ],
['./exp/log/Saved/21_10_01/model9_100/model9_100_replicate_0/2019-10-11--21:41:4623572/logs.mat', ],
]

relu50 = [
['./exp/log/Saved/21_10_01/model0_50/model0_50_replicate_0/2019-10-11--22:00:4265304/logs.mat', ],
['./exp/log/Saved/21_10_01/model1_50/model1_50_replicate_0/2019-10-11--22:00:4291347/logs.mat', ],
['./exp/log/Saved/21_10_01/model2_50/model2_50_replicate_0/2019-10-11--22:00:424533/logs.mat', ],
['./exp/log/Saved/21_10_01/model3_50/model3_50_replicate_0/2019-10-11--22:00:4264266/logs.mat', ],
['./exp/log/Saved/21_10_01/model4_50/model4_50_replicate_0/2019-10-11--22:20:5997659/logs.mat', ],
['./exp/log/Saved/21_10_01/model5_50/model5_50_replicate_0/2019-10-11--22:21:157735/logs.mat', ],
['./exp/log/Saved/21_10_01/model6_50/model6_50_replicate_0/2019-10-11--22:21:3973428/logs.mat', ],
['./exp/log/Saved/21_10_01/model7_50/model7_50_replicate_0/2019-10-11--22:23:0154889/logs.mat', ],
['./exp/log/Saved/21_10_01/model8_50/model8_50_replicate_0/2019-10-11--22:27:3141613/logs.mat', ],
['./exp/log/Saved/21_10_01/model9_50/model9_50_replicate_0/2019-10-11--22:42:2522925/logs.mat', ],
]

lrelu200 = [
['./exp/log/Saved/21_10_02/model_lrelu_0_200/model_lrelu_0_200_replicate_0/2019-10-21--11:04:5544962/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_1_200/model_lrelu_1_200_replicate_0/2019-10-21--11:04:5558806/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_2_200/model_lrelu_2_200_replicate_0/2019-10-21--11:04:5949598/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_3_200/model_lrelu_3_200_replicate_0/2019-10-21--11:05:0135843/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_4_200/model_lrelu_4_200_replicate_0/2019-10-21--11:05:0031098/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_5_200/model_lrelu_5_200_replicate_0/2019-10-21--11:05:0036726/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_6_200/model_lrelu_6_200_replicate_0/2019-10-21--11:05:0013519/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_7_200/model_lrelu_7_200_replicate_0/2019-10-21--11:05:0035589/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_8_200/model_lrelu_8_200_replicate_0/2019-10-21--11:08:4437650/logs.mat', ],
['./exp/log/Saved/21_10_02/model_lrelu_9_200/model_lrelu_9_200_replicate_0/2019-10-21--11:08:4479719/logs.mat', ],
]

relu_suite_200 = [
['./exp/log/Saved/21_10_02/model_10_200/model_10_200_replicate_0/2019-10-21--11:08:4471830/logs.mat', ],
['./exp/log/Saved/21_10_02/model_11_200/model_11_200_replicate_0/2019-10-21--11:08:4414694/logs.mat', ],
['./exp/log/Saved/21_10_02/model_12_200/model_12_200_replicate_0/2019-10-21--11:08:5062907/logs.mat', ],
['./exp/log/Saved/21_10_02/model_13_200/model_13_200_replicate_0/2019-10-21--11:08:5018682/logs.mat', ],
['./exp/log/Saved/21_10_02/model_14_200/model_14_200_replicate_0/2019-10-21--11:08:5020192/logs.mat', ],
['./exp/log/Saved/21_10_02/model_15_200/model_15_200_replicate_0/2019-10-21--11:08:5045128/logs.mat', ],
['./exp/log/Saved/21_10_02/model_16_200/model_16_200_replicate_0/2019-10-21--11:08:5021988/logs.mat', ],
['./exp/log/Saved/21_10_02/model_17_200/model_17_200_replicate_0/2019-10-21--11:08:5018393/logs.mat', ],
['./exp/log/Saved/21_10_02/model_18_200/model_18_200_replicate_0/2019-10-21--11:08:5028832/logs.mat', ],
['./exp/log/Saved/21_10_02/model_19_200/model_19_200_replicate_0/2019-10-21--11:08:5049399/logs.mat', ],
]

# 22/10
dropout200 = [
['./exp/log/Saved/22_10_01/model_dropout_0_200/model_dropout_0_200_replicate_0/2019-10-21--16:59:4472436/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_1_200/model_dropout_1_200_replicate_0/2019-10-21--16:59:4492807/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_2_200/model_dropout_2_200_replicate_0/2019-10-21--16:59:4786858/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_3_200/model_dropout_3_200_replicate_0/2019-10-21--16:59:5081009/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_4_200/model_dropout_4_200_replicate_0/2019-10-21--16:59:5035929/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_5_200/model_dropout_5_200_replicate_0/2019-10-21--16:59:5011171/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_6_200/model_dropout_6_200_replicate_0/2019-10-21--16:59:5079306/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_7_200/model_dropout_7_200_replicate_0/2019-10-21--16:59:5070437/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_8_200/model_dropout_8_200_replicate_0/2019-10-21--16:59:5052495/logs.mat', ],
['./exp/log/Saved/22_10_01/model_dropout_9_200/model_dropout_9_200_replicate_0/2019-10-21--17:00:0525395/logs.mat', ],
]

onrack200 = [
['./exp/log/Saved/22_10_01/model_onrack_0_200/model_onrack_0_200_replicate_0/2019-10-21--16:24:3554411/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_1_200/model_onrack_1_200_replicate_0/2019-10-21--16:24:3556260/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_2_200/model_onrack_2_200_replicate_0/2019-10-21--16:24:3526819/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_3_200/model_onrack_3_200_replicate_0/2019-10-21--16:24:3559624/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_4_200/model_onrack_4_200_replicate_0/2019-10-21--16:24:3567876/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_5_200/model_onrack_5_200_replicate_0/2019-10-21--16:24:3536854/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_6_200/model_onrack_6_200_replicate_0/2019-10-21--16:24:3566847/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_7_200/model_onrack_7_200_replicate_0/2019-10-21--16:24:3550471/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_8_200/model_onrack_8_200_replicate_0/2019-10-21--16:24:3528869/logs.mat', ],
['./exp/log/Saved/22_10_01/model_onrack_9_200/model_onrack_9_200_replicate_0/2019-10-21--16:24:353355/logs.mat', ],
]

torqueless200 = [
['./exp/log/Saved/22_10_01/model_torqueless_0_200/model_torqueless_0_200_replicate_0/2019-10-21--16:33:1018788/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_1_200/model_torqueless_1_200_replicate_0/2019-10-21--16:33:1056780/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_2_200/model_torqueless_2_200_replicate_0/2019-10-21--16:33:1044842/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_3_200/model_torqueless_3_200_replicate_0/2019-10-21--16:33:1022807/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_4_200/model_torqueless_4_200_replicate_0/2019-10-21--16:33:0964086/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_5_200/model_torqueless_5_200_replicate_0/2019-10-21--16:33:1038843/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_6_200/model_torqueless_6_200_replicate_0/2019-10-21--16:33:1065736/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_7_200/model_torqueless_7_200_replicate_0/2019-10-21--16:33:1028196/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_8_200/model_torqueless_8_200_replicate_0/2019-10-21--16:33:2513569/logs.mat', ],
['./exp/log/Saved/22_10_01/model_torqueless_9_200/model_torqueless_9_200_replicate_0/2019-10-21--16:36:5471912/logs.mat', ],
]
