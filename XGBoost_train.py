import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier as XGBoostClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
import pickle

#Parameters
BEST_FEATURE = True
train = True
data_path = "data_complex.xlsx"
dst_path = ""


df = pd.read_excel(data_path)

df["coded_label"] = df["LABEL"].apply(lambda x: 1 if x == "ACTIVE" else 0)

prova = df.drop(columns=["LABEL", "coded_label"])

if BEST_FEATURE:
    best_feature = [2, 4, 19, 21, 34, 36, 43, 53, 55, 56, 59, 60, 67, 145, 200, 237, 309, 332, 333, 334, 335, 339, 348, 352, 363, 366, 387, 394, 397, 405, 406, 407, 411, 514, 587, 622, 807, 809, 815, 816, 817, 818, 821, 823, 824, 827, 828, 829, 830, 831, 833, 835, 838, 839, 840, 843, 844, 847, 849, 850, 852, 853, 854, 855, 856, 857, 858, 859, 860, 862, 863, 864, 865, 866, 867, 869, 870, 871, 872, 873, 878, 879, 880, 881, 882, 884, 885, 887, 889, 890, 891, 892, 895, 898, 899, 900, 901, 902, 910, 913, 917, 918, 919, 920, 921, 922, 923, 924, 931, 935, 939, 943, 945, 948, 956, 957, 965, 967, 974, 975, 983, 985, 1000, 1003, 1004, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1063, 1064, 1065, 1066, 1067, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1095, 1096, 1097, 1098, 1102, 1107, 1109, 1116, 1119, 1120, 1121, 1122, 1123, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1140, 1141, 1142, 1144, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1171, 1172, 1173, 1174, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1200, 1201, 1202, 1203, 1205, 1206, 1207, 1208, 1209, 1210, 1212, 1213, 1214, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1233, 1234, 1235, 1236, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1247, 1248, 1249, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1266, 1267, 1268, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1283, 1284, 1287, 1288, 1289, 1290, 1293, 1294, 1295, 1297, 1298, 1299, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1311, 1312, 1314, 1316, 1317, 1320, 1321, 1322, 1323, 1324, 1327, 1328, 1332, 1333, 1335, 1339, 1341, 1342, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1356, 1357, 1361, 1362, 1363, 1365, 1369, 1370, 1371, 1372, 1375, 1376, 1377, 1378, 1379, 1382, 1383, 1384, 1388, 1389, 1390, 1391, 1392, 1412, 1415, 1416, 1417, 1418, 1420, 1421, 1422, 1424, 1426, 1428, 1429, 1430, 1438, 1441, 1444, 1445, 1447, 1449, 1461, 1466, 1467, 1468, 1477, 1480, 1481, 1484, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1517, 1522, 1529, 1545, 1547, 1551, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1581, 1583, 1584, 1586, 1588, 1591, 1592, 1594, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1638, 1639, 1640, 1641, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1656, 1684, 1687, 1688, 1689, 1690, 1696, 1706, 1708, 1709, 1716]
    filtered_best = prova.iloc[:, best_feature]
else:
    short_feat = [
        4, 19, 21, 55, 56, 59, 67, 309, 332, 387, 394, 397, 809, 815, 816, 817, 818, 823, 831, 835, 838, 839, 840, 850, 854, 855, 857, 859, 860, 862, 872, 873, 880, 881, 892, 910, 913, 917, 918, 919, 921, 922, 923, 924, 931, 935, 943, 945, 1009, 1010, 1011, 1012, 1013, 1045, 1065, 1072, 1073, 1074, 1075, 1076, 1090, 1091, 1109, 1122, 1123, 1128, 1130, 1131, 1132, 1134, 1135, 1136, 1141, 1142, 1151, 1152, 1153, 1160, 1163, 1164, 1165, 1166, 1167, 1168, 1171, 1172, 1173, 1174, 1176, 1177, 1178, 1179, 1180, 1181, 1185, 1187, 1193, 1194, 1195, 1196, 1202, 1205, 1206, 1209, 1212, 1219, 1223, 1224, 1225, 1226, 1227, 1228, 1233, 1235, 1238, 1239, 1240, 1241, 1242, 1243, 1253, 1254, 1255, 1258, 1259, 1260, 1262, 1263, 1274, 1277, 1278, 1299, 1303, 1306, 1311, 1312, 1314, 1323, 1324, 1342, 1349, 1353, 1361, 1365, 1371, 1376, 1391, 1392, 1468, 1486, 1488, 1489, 1490, 1492, 1493, 1494, 1495, 1496, 1497, 1522, 1561, 1562, 1563, 1564, 1565, 1583, 1603, 1607, 1610, 1611, 1612, 1633, 1634, 1635, 1644, 1646, 1647, 1648, 1649, 1650
    ]
    filtered_best = prova.iloc[:, short_feat]


x, y = filtered_best.values, df["coded_label"].values

for i in x.columns:
    if x[i].dtype == 'object':
        x[i] = x[i].str.replace('na', '0').astype(float)  # Replace 'None' with '0'



skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

f1_list, bal_list, prec_list, recall_list = [], [], [], []
for i, (train_index, test_index) in enumerate(tqdm(skf.split(x, y), total=10)):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #! check if xgboost is the best model. AFTER SFS
    xgb = XGBoostClassifier(
        device='cuda', verbose=2, max_depth=8,
        n_estimators=1000, eval_metric="auc",
        booster='dart', eta=0.05, subsample=1, scale_pos_weight=3, random_state=17
    ) 

    if train:
        xgb_trained = xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)])

        xgb.get_booster().save_model(f"{dst_path}/xgboost_{i+1}.json")

    else:
        booster = xgb.Booster()
        booster.load_model(f"{dst_path}/xgboost_{i+1}.json")


    score = xgb_trained.score(x_test, y_test)
    print(f"Fold score: {score}")
    y_pred = xgb_trained.predict(x_test)


    f1 = f1_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_list.append(f1)
    bal_list.append(balanced_acc)
    prec_list.append(precision)
    recall_list.append(recall)
    print(f"F1 Score: {f1}")
    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    file = open(f"{dst_path}/xgb_model_results.csv", "a")
    if i == 0:
        file.writelines("Fold,F1 Score,Balanced Accuracy,Precision,Recall\n")
    file.writelines(f"{i+1},{f1},{balanced_acc},{precision},{recall}\n")
    file.close()


print(f"Average F1 Score: {sum(f1_list) / len(f1_list)}")
print(f"Average Balanced Accuracy: {sum(bal_list) / len(bal_list)}")
print(f"Average Precision: {sum(prec_list) / len(prec_list)}")
print(f"Average Recall: {sum(recall_list) / len(recall_list)}")

Average_F1 = sum(f1_list) / len(f1_list)
Average_bal = sum(bal_list) / len(bal_list)
Average_prec = sum(prec_list) / len(prec_list)
Average_rec = sum(recall_list) / len(recall_list)

if train:
    file = open(f"{dst_path}/xgb_model_results.csv", "a")
    file.writelines(
        "Type,Average F1 Score,Average Balanced Accuracy,Average Precision,Average Recall\n")
    file.writelines(
        f"average,{Average_F1},{Average_bal},{Average_prec},{Average_rec}\n")
    file.close()
else:
    file = open(f"{dst_path}/xgb_model_results.csv", "a")
    file.writelines(f"{'*'*100}\n")
    file.writelines(
        "Type,Average F1 Score,Average Balanced Accuracy,Average Precision,Average Recall\n")
    file.writelines(
        f"average,{Average_F1},{Average_bal},{Average_prec},{Average_rec}\n")
    file.close()
