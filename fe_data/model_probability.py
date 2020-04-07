# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:56:49 2018

@author: yifei_lin
"""
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

#特征值转换成woe
def AssignGroupWOE(x, cutOff, woe):
    N = len(cutOff)
    if x <= min(cutOff):
        return woe[0]
    elif x > max(cutOff):
        return woe[-1]
    else:
        for i in range(N - 1):
            if cutOff[i] < x <= cutOff[i + 1]:
                return woe[i + 1]

#入模特征字段、英文名、中文名
keepFeaturesFinal=['sjtu_step4_sample_ordernum_idkey_borrbasic1_14907', 'sjtu_step4_sample_ordernum_idkey_borrbasic2_11718', 'sjtu_step4_sample_ordernum_idkey_borrex_993', 'sjtu_step4_sample_ordernum_idkey_borrex_729', 'sjtu_step4_sample_ordernum_idkey_borrex_753', 'sjtu_step4_sample_ordernum_idkey_borrex_1002', 'sjtu_step4_sample_ordernum_idkey_borrex_913', 'sjtu_step4_sample_ordernum_idkey_borrex_3678', 'sjtu_step4_sample_ordernum_idkey_borrex_953', 'sjtu_step4_sample_ordernum_idkey_borrex_332', 'sjtu_step4_sample_ordernum_idkey_payex_2616', 'sjtu_step4_sample_ordernum_idkey_payex_71', 'sjtu_step4_sample_ordernum_idkey_payex_344', 'sjtu_step4_sample_ordernum_idkey_payex_492', 'sjtu_step4_sample_ordernum_idkey_payex_532', 'sjtu_step4_sample_ordernum_idkey_payex_2508', 'sjtu_step4_sample_ordernum_idkey_payex_61', 'sjtu_step4_sample_ordernum_idkey_payex_622', 'sjtu_step4_sample_ordernum_idkey_payex_925', 'sjtu_step4_sample_ordernum_idkey_payex_769', 'sjtu_step4_sample_ordernum_idkey_payex_1760', 'sjtu_step4_sample_ordernum_idkey_payex_3417', 'sjtu_step4_sample_ordernum_idkey_payex_1577', 'sjtu_step4_sample_ordernum_idkey_paybasic_20625', 'sjtu_step4_sample_ordernum_idkey_paybasic_5226', 'sjtu_step4_sample_ordernum_idkey_paybasic_2149', 'sjtu_step4_sample_ordernum_idkey_paybasic_642', 'sjtu_step4_sample_ordernum_idkey_paybasic_13463', 'sjtu_step4_sample_ordernum_idkey_paybasic_1278', 'sjtu_step4_sample_ordernum_idkey_paybasic_16866', 'sjtu_step4_sample_ordernum_idkey_paybasic_430', 'sjtu_step4_sample_ordernum_idkey_paybasic_621', 'sjtu_step4_sample_ordernum_idkey_paybasic_382', 'sjtu_step4_sample_ordernum_idkey_paybasic_115', 'sjtu_step4_sample_ordernum_idkey_paybasic_15305', 'sjtu_step4_sample_ordernum_idkey_paybasic_179', 'sjtu_step4_sample_ordernum_idkey_paybasic_14715', 'sjtu_step4_sample_ordernum_idkey_paybasic_20745', 'sjtu_step4_sample_ordernum_idkey_paybasic_13628', 'sjtu_step4_sample_ordernum_idkey_paybasic_16', 'sjtu_step4_sample_ordernum_idkey_paybasic_16327', 'sjtu_step4_sample_ordernum_idkey_paybasic_126']                   
keepFeaturesFinal_en=['std_alltime_noncdq_sureduelend_days_m3', 'new_std_alltime_allpro_sureduelend_days_m6', 'per_alltime_cq_succlenddivlend_cnt_d7', 'per_week_allpro_succlenddivlend_cnt_d7', 'per_week_allpro_succlenddivlend_cnt_cnt3', 'per_alltime_cqdivallpro_succlend_cnt_m1', 'per_alltime_dq_succlenddivlend_cnt_m1', 'new_per_sum_work_cdqpro_likeduelend_days_m3', 'per_alltime_zq_succlenddivlend_cnt_d15', 'per_work_dqdivallpro_lend_amt_d7', 'new_per_sum_work_zq_pay_amt_d7', 'per_weekdivalltime_cdqpro_pay_amt_d15', 'per_work_dqdivallpro_pay_amt_m3', 'per_alltime_cqdivallpro_pay_cnt_m3', 'per_week_cqdivallpro_pay_amt_cnt3', 'new_per_sum_work_cdqpro_pay_amt_d7', 'per_alltime_cdqprodivallpro_pay_amt_cnt10', 'per_week_xfjrdivallpro_pay_amt_d15', 'per_alltime_dq_succpaydivpay_cnt_m12', 'per_week_allpro_succpaydivpay_cnt_cnt30', 'kqper_sum_week_dq_pay_amt_d15_m1', 'new_per_sum_work_zq_likeduepay_days_m3', 'per_alltime_noncdqdivallpro_likeduepay_days_m3', 'new_std_week_allpro_likeduepay_days_m1', 'dis_alltime_noncdq_succpay_mindis_m3', 'dis_work_dq_pay_mindis_m1', 'dis_alltime_cdqpro_pay_mindis_m3', 'std_week_cdqpro_likeduepay_days_cnt30', 'std_alltime_noncdq_pay_amt_cnt5', 'new_sum_alltime_noncdq_succpay_month_m3', 'dis_work_allpro_pay_mindis_m1', 'dis_alltime_cdqpro_pay_mindis_m1', 'dis_week_allpro_pay_mindis_cnt30', 'min_alltime_allpro_pay_amt_cnt3', 'new_std_alltime_dq_pay_tperiod_m3', 'min_alltime_allpro_pay_amt_cnt30', 'new_avg_work_allpro_pay_amt_m12', 'new_std_work_cdqpro_likeduepay_days_m1', 'std_week_noncdq_likeduepay_days_cnt30', 'dis_alltime_allpro_pay_mindis_d7', 'new_sum_alltime_allpro_succpay_membercnt_m1', 'dis_alltime_allpro_pay_maxdis_cnt3']
keepFeaturesFinal_cn=['最近3个月全部时间非超短期现金贷方差借贷确定逾期天数', '最近半年新增平台全部时间全部产品方差借贷确定逾期天数', '最近7天全部时间长期现金贷非逾期借贷在总借贷中订单数占比', '最近7天非工作日全部产品非逾期借贷在总借贷中订单数占比', '最近3次非工作日全部产品非逾期借贷在总借贷中订单数占比', '最近1个月全部时间非超短期现金贷在全部产品中非逾期借贷订单数占比', '最近1个月全部时间短期现金贷非逾期借贷在总借贷中订单数占比', '最近3个月新增平台工作日超短期现金贷累计借贷疑似逾期天数占比', '最近15天全部时间中期现金贷非逾期借贷在总借贷中订单数占比', '最近7天工作日短期现金贷在全部产品中借贷金额占比', '最近7天新增平台工作日中期现金贷累计还款金额占比', '最近15天非工作日在全部时间中超短期现金贷还款金额占比', '最近3个月工作日短期现金贷在全部产品中还款金额占比', '最近3个月全部时间非超短期现金贷在全部产品中还款订单数占比', '最近3次非工作日非超短期现金贷在全部产品中还款金额占比', '最近7天新增平台工作日超短期现金贷累计还款金额占比', '最近10次全部时间超短期现金贷在全部产品中还款金额占比', '最近15天非工作日长期现金贷在全部产品中还款金额占比', '最近12个月全部时间短期现金贷非逾期还款在总还款中订单数占比', '最近30次非工作日全部产品非逾期还款在总还款中订单数占比', '最近半月非工作日短期现金贷还款金额跨期占比', '最近3个月新增平台工作日中期现金贷累计还款疑似逾期天数占比', '最近3个月全部时间中期现金贷在全部产品中还款疑似逾期天数占比', '最近一月新增平台非工作日全部产品方差还款疑似逾期天数', '最近3个月全部时间非超短期现金贷距离非逾期还款最后一次距今天数', '最近1个月工作日短期现金贷距离还款最后一次距今天数', '最近3个月全部时间超短期现金贷距离还款最后一次距今天数', '最近30次非工作日超短期现金贷方差还款疑似逾期天数', '最近5次全部时间非超短期现金贷方差还款金额', '最近3个月新增平台全部时间非超短期现金贷累计非逾期还款月数', '最近1个月工作日全部产品距离还款最后一次距今天数', '最近1个月全部时间超短期现金贷距离还款最后一次距今天数', '最近30次非工作日全部产品距离还款最后一次距今天数', '最近3次全部时间全部产品最小还款金额', '最近3个月新增平台全部时间短期现金贷方差还款时间间隔', '最近30次全部时间全部产品最小还款金额', '最近一年新增平台工作日全部产品平均还款金额', '最近一月新增平台工作日超短期现金贷方差还款疑似逾期天数', '最近30次非工作日非超短期现金贷方差还款疑似逾期天数', '最近7天全部时间全部产品距离还款最后一次距今天数', '最近一月新增平台全部时间全部产品累计非逾期还款平台数', '最近3次全部时间全部产品距离还款第一次距今天数']

#特征字段分箱cutoff
cutoff={'sjtu_step4_sample_ordernum_idkey_borrbasic1_14907': [-99998.0, 0.0, 2.8284, 4.5, 7.5, 10.3387, 13.0213, 21.5], 
        'sjtu_step4_sample_ordernum_idkey_borrbasic2_11718': [-99998.0, 2.245, 9.5336], 
        'sjtu_step4_sample_ordernum_idkey_borrex_993': [-999978.0, -99998.0, 33.3333, 50.0, 75.0], 
        'sjtu_step4_sample_ordernum_idkey_borrex_729': [-999978.0, -99998.0, 25.0, 33.3333, 50.0, 87.5], 
        'sjtu_step4_sample_ordernum_idkey_borrex_753': [-999978.0, -99998.0, 33.3333, 50.0, 66.6667], 
        'sjtu_step4_sample_ordernum_idkey_borrex_1002': [-999978.0, 13.6364, 23.5294, 31.25, 40.0], 
        'sjtu_step4_sample_ordernum_idkey_borrex_913': [-999978.0, -99998.0, 25.0, 40.0, 50.0, 83.3333], 
        'sjtu_step4_sample_ordernum_idkey_borrex_3678': [-999978.0, -99998.0, 51.6129, 95.2381], 
        'sjtu_step4_sample_ordernum_idkey_borrex_953': [-999978.0, -99998.0, 20.0, 28.5714, 40.0, 42.8571], 
        'sjtu_step4_sample_ordernum_idkey_borrex_332': [-999978.0, 14.6991, 50.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_2616': [-999978.0, -999977.0, -999976.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_71': [-999977.0, -999976.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_344': [-999977.0, -99998.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_492': [-99998.0, 4.1667, 6.8182, 8.6957, 15.5556, 20.5128, 24.0, 26.6667, 30.7692, 31.4286, 37.5, 47.0588, 47.3684], 
        'sjtu_step4_sample_ordernum_idkey_payex_532': [-999977.0, -999976.0, 20.3356], 
        'sjtu_step4_sample_ordernum_idkey_payex_2508': [-999977.0, -999976.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_61': [-999977.0, 0.0, 63.0179, 98.8423], 
        'sjtu_step4_sample_ordernum_idkey_payex_622': [-999977.0, -999976.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_925': [-999978.0, -99998.0, 14.2857, 16.6667, 20.0, 25.0, 33.3333, 66.6667, 93.3333], 
        'sjtu_step4_sample_ordernum_idkey_payex_769': [-99998.0, 20.0, 29.4118, 37.931, 46.1538, 96.6667], 
        'sjtu_step4_sample_ordernum_idkey_payex_1760': [-999978.0, -999977.0, -999976.0, 0.0], 
        'sjtu_step4_sample_ordernum_idkey_payex_3417': [-999978.0, 1.9231, 10.0, 18.1818, 48.2759, 99.2], 
        'sjtu_step4_sample_ordernum_idkey_payex_1577': [13.6364, 27.4809, 45.2381, 73.8095], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_20625': [-99998.0, 0.0, 2.5, 5.3541, 11.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_5226': [-99998.0, 1.0, 3.0, 6.0, 10.0, 17.0, 28.0, 50.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_2149': [-99998.0, 1.0, 20.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_642': [1.0, 2.0, 32.0, 51.0, 71.0, 83.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_13463': [-99998.0, 6.4, 14.0, 26.5], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_1278': [0.0, 256.1236, 442.968], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_16866': [-99998.0, 1.0, 2.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_430': [1.0, 14.0, 21.0, 25.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_621': [-99998.0, 1.0, 10.0, 16.0, 23.0, 25.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_382': [2.0, 3.0, 4.0, 5.0, 11.0, 28.0, 55.0, 79.0, 104.0, 141.0, 197.0, 268.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_115': [0.0, 180.0, 280.78, 509.3, 740.57], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_15305': [-99998.0, 3.4187, 3.5, 5.5202, 10.5], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_179': [0.0, 108.8, 193.71, 380.77], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_14715': [0.0, 48.405, 110.6667, 224.0, 268.01, 350.0, 426.0, 499.6667, 1127.962, 1526.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_20745': [-99998.0, 0.0, 1.2472, 2.5, 4.3301, 11.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_13628': [-99998.0, 2.8284, 4.5613, 7.7942, 12.51, 14.3836], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_16': [-99998.0, 1.0, 3.0, 5.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_16327': [-99998.0, 1.0, 2.0, 3.0], 
        'sjtu_step4_sample_ordernum_idkey_paybasic_126': [1.0, 5.0, 7.0, 10.0, 12.0, 17.0, 23.0]}

#特征字段分箱woe
woe={'sjtu_step4_sample_ordernum_idkey_borrbasic1_14907': [0.165, -0.6458, -0.9617, -1.3595, -1.4433, -1.7205, -1.8426, -2.8463, -4.8421], 
     'sjtu_step4_sample_ordernum_idkey_borrbasic2_11718': [0.4861, -0.2866, -0.7255, -1.2616], 
     'sjtu_step4_sample_ordernum_idkey_borrex_993': [-0.0452, -2.4137, -3.0843, -1.5528, -1.1384, 0.7298], 
     'sjtu_step4_sample_ordernum_idkey_borrex_729': [0.0853, -1.6888, -2.7558, -1.7384, -0.9328, -0.6412, 0.3244], 
     'sjtu_step4_sample_ordernum_idkey_borrex_753': [0.1819, -2.2887, -1.6457, -0.997, -0.6573, 0.5057], 
     'sjtu_step4_sample_ordernum_idkey_borrex_1002': [-1.8405, -0.0873, 0.5379, 0.5887, 0.7143, 0.5985], 
     'sjtu_step4_sample_ordernum_idkey_borrex_913': [0.3583, -2.2396, -3.4763, -2.0122, -1.1189, -0.6853, 0.6215], 
     'sjtu_step4_sample_ordernum_idkey_borrex_3678': [0.4099, -1.0239, -2.2138, -2.1532, -1.5327], 
     'sjtu_step4_sample_ordernum_idkey_borrex_953': [0.1481, -1.7698, -3.842, -3.2003, -2.3567, -0.3762, 0.2692], 
     'sjtu_step4_sample_ordernum_idkey_borrex_332': [0.5085, 0.0339, -0.7654, -0.8056], 
     'sjtu_step4_sample_ordernum_idkey_payex_2616': [0.091, -1.3249, -1.2983, 0.621], 
     'sjtu_step4_sample_ordernum_idkey_payex_71': [-0.3344, -2.0529, 0.3445], 
     'sjtu_step4_sample_ordernum_idkey_payex_344': [-2.6127, 0.3493, -0.1123], 
     'sjtu_step4_sample_ordernum_idkey_payex_492': [-0.2587, -0.381, -0.3179, -0.3004, -0.0237, 0.2268, 0.3794, 0.4285, 0.5898, 0.6138, 0.8149, 0.8343, 0.8475, 1.1964], 
     'sjtu_step4_sample_ordernum_idkey_payex_532': [-0.265, -2.2169, 0.1667, 0.4212], 
     'sjtu_step4_sample_ordernum_idkey_payex_2508': [-0.0547, -1.5964, 0.3229], 
     'sjtu_step4_sample_ordernum_idkey_payex_61': [-4.7042, -0.4007, 0.4963, 0.2367, -0.7324], 
     'sjtu_step4_sample_ordernum_idkey_payex_622': [0.0447, -1.5042, 0.4156], 
     'sjtu_step4_sample_ordernum_idkey_payex_925': [0.3704, -1.3733, -1.7431, -1.6899, -1.4419, -1.0726, -0.6581, -0.078, 0.4133, 0.7155], 
     'sjtu_step4_sample_ordernum_idkey_payex_769': [-1.0325, -2.0195, -1.5028, -1.2092, -1.2029, -0.0288, 0.9708],
     'sjtu_step4_sample_ordernum_idkey_payex_1760': [0.0567, -1.1427, -1.4367, 0.5086, 0.656], 
     'sjtu_step4_sample_ordernum_idkey_payex_3417': [0.3575, -0.2657, -1.4194, -1.578, -1.4547, -0.8458, -0.3827], 
     'sjtu_step4_sample_ordernum_idkey_payex_1577': [0.5525, -0.8468, -0.9281, -0.5948, -0.038], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_20625': [0.1472, -0.749, -2.0646, -2.2611, -2.5301, -2.4467], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_5226': [-0.8697, 0.8998, 0.6921, 0.6838, 0.4794, 0.4178, 0.4159, 0.182, -0.1197], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_2149': [0.3165, -1.6658, 0.0803, 0.1383],
     'sjtu_step4_sample_ordernum_idkey_paybasic_642': [-0.7133, -0.4953, 0.0587, 0.2814, 0.3529, 0.5473, 0.7191], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_13463': [0.4059, -0.4589, -1.2261, -1.479, -1.6471], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_1278': [-0.9296, -0.0004, 0.312, 0.2733], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_16866': [-0.4446, -0.0451, 0.3135, 0.6458], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_430': [-1.1226, 0.2557, 0.5729, 0.6978, 0.836], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_621': [-0.2968, -1.0637, 0.0855, 0.2851, 0.3438, 0.5035, 0.5095], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_382': [-0.3439, -0.9064, -1.4251, -1.0585, -0.3847, -0.1285, 0.0633, 0.2495, 0.4227, 0.5771, 0.669, 0.7353, 0.9307], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_115': [-0.7634, 0.7879, 1.0489, 1.0459, 0.8391, 0.7916], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_15305': [0.2517, -0.263, -1.2274, -1.6792, -0.7394, -0.4912], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_179': [-0.1975, 1.2497, 1.4132, 1.5489, 1.6031], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_14715': [-1.8624, -2.5735, -1.842, -1.3683, -1.333, -1.1104, -0.9365, -0.9231, -0.0395, 0.4972, 0.9851], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_20745': [0.2074, -0.7009, -2.1804, -2.3398, -2.3132, -2.1203, -2.1027], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_13628': [0.414, -0.2341, -1.9596, -2.1479, -1.9603, -1.8621, -1.6236], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_16': [0.6318, -0.8911, 0.2889, 0.5058, 0.5886], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_16327': [-0.57, 0.3435, 0.4221, 0.3844, 0.3282], 
     'sjtu_step4_sample_ordernum_idkey_paybasic_126': [-1.1388, 0.2381, 0.392, 0.42, 0.5339, 0.5378, 0.4714, 0.4524]}

#特征值对应系数
lrModel_coefficients=[0.0336, -0.3679, -0.2479, 0.0352, -0.1329, -0.265, -0.3211, -0.2935, -0.2497, -0.0819, -0.1808, -0.2193, -0.1141, -0.1689, -0.1, -0.0731, -0.2097, -0.0801, -0.1034, 0.0703, -0.0602, -0.092, -0.0555, -0.124, -0.0862, -0.0677, 0.1006, -0.2769, -0.0872, -0.2191, -0.088, -0.0721, -0.1607, -0.1191, -0.0947, -0.2259, -0.1058, -0.1838, -0.0692, -0.135, 0.0598, -0.2134]
lrModel_intercept=-0.6808373075147756

#读取数据集
data=pd.read_csv('sjtu_step4_sample_ordernum_keepfeaturesfinal_train.csv',names=['id_card_md5','name_md5','lend_day','target']+keepFeaturesFinal)

#特征值转换成woe
data_woe=pd.DataFrame()
for i in keepFeaturesFinal:
    data_woe[i]=data[i].apply(lambda x: AssignGroupWOE(x, cutoff[i], woe[i]))

#由woe计算好样本概率
data_woe['prob']=data_woe[keepFeaturesFinal].apply(lambda x: 1-(1 / (1 + np.exp(-(sum(lrModel_coefficients * x) + lrModel_intercept)))),axis=1)


        
        
        
        
        
        