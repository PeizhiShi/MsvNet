
import libs.multi as mlt


group_idx = 3

p,r = mlt.test_msvnet(group_idx)

print('Precision for the MsvNet on group ', group_idx, ': ',p)
print('Recall for the MsvNet on group ', group_idx, ': ',r)

p,r = mlt.test_featurenet(group_idx)


print('Precision for the FeatureNet on group ', group_idx, ': ',p)
print('Recall for the FeatureNet on group ', group_idx, ': ',r)

