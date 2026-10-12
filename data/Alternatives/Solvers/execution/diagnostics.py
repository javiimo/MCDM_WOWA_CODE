"""Retain scientific inputs and detailed diagnostics without spatial assumptions."""
import hashlib
import json
import numpy as np
from .storage import atomic_json


def retain(path, instance, starts, risk, mean, quantile, resources, objective):
    horizon=instance['T'];mean=np.asarray(mean,dtype=float);quantile=np.asarray(quantile,dtype=float)
    excess=np.maximum(quantile-mean,0)
    activities=[[] for _ in range(horizon)];intervals={};contributions={}
    for name,start in starts.items():
        intervention=instance['Interventions'][name];duration=int(intervention['Delta'][start-1])
        intervals[name]={'start':start,'duration':duration,'end_exclusive':start+duration,'latest_start':int(intervention['tmax'])}
        profile=[0.0]*horizon
        for t in range(start,start+duration):
            activities[t-1].append(name)
            values=intervention['risk'][str(t)][str(start)]
            profile[t-1]=sum(float(v) for v in values)/len(values)
        contributions[name]=profile
    loads={};bottleneck=[[] for _ in range(horizon)]
    for name,usage in resources.items():
        bounds=instance['Resources'][name];upper=np.asarray(bounds['max'],dtype=float);lower=np.asarray(bounds['min'],dtype=float);usage=np.asarray(usage,dtype=float)
        normalized=np.divide(np.maximum(upper-usage,0),upper-lower,out=np.zeros(horizon),where=upper>lower)
        for t,slack in enumerate(normalized):bottleneck[t].append(float(slack))
        loads[name]={'usage':usage.tolist(),'lower_bound':lower.tolist(),'upper_bound':upper.tolist(),'lower_slack':(usage-lower).tolist(),'upper_slack':(upper-usage).tolist(),'normalized_upper_slack':normalized.tolist(),'binding_upper_periods':[i+1 for i,v in enumerate(upper-usage) if abs(v)<=1e-8],'binding_lower_periods':[i+1 for i,v in enumerate(usage-lower) if abs(v)<=1e-8]}
    profile_slack=[min(v) if v else 0 for v in bottleneck]
    schedule_hash=hashlib.sha256(json.dumps(sorted(starts.items()),separators=(',',':')).encode()).hexdigest()
    data={'schema':2,'objectives':dict(objective,weighted_mean_component=objective['alpha']*objective['mean_risk'],weighted_excess_component=(1-objective['alpha'])*objective['expected_excess']),
          'instance':{'T':horizon,'Quantile':instance['Quantile'],'Alpha':instance['Alpha'],'scenario_counts':instance['Scenarios_number'],'interventions':len(starts),'resources':len(loads),'exclusions':len(instance['Exclusions']),'seasons':instance['Seasons']},
          'risk':{'scenario_values_in_original_order':[[float(v) for v in row] for row in risk],'scenario_identity_warning':'Indices are local to each period; cross-period scenario identity is not asserted.','mean_by_period':mean.tolist(),'quantile_by_period':quantile.tolist(),'positive_excess_by_period':excess.tolist(),'variance_by_period':[float(np.var(row)) for row in risk],'minimum_by_period':[min(row) for row in risk],'maximum_by_period':[max(row) for row in risk],'intervention_mean_contributions':contributions},
          'schedule':{'starts':starts,'intervals':intervals,'active_by_period':activities,'active_counts':[len(x) for x in activities],'fingerprint_sha256':schedule_hash},
          'resources':loads,'bottleneck_slack_by_period':profile_slack,
          'definitions':{'objective':'Alpha*mean_t(mean_s(risk[t,s])) + (1-Alpha)*mean_t(max(Q[t]-mean[t],0))','quantile':'official empirical order statistic ceil(scenario_count*Quantile), not interpolated percentile','normalized_upper_slack':'max(upper-usage,0)/(upper-lower), zero for degenerate intervals'}}
    atomic_json(path,data)
    return data
