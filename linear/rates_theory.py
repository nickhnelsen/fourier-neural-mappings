alpha = 4.5
beta = 1.5
r = -1.5 # -1.5, -0.5, 0.5

# Assumptions
if alpha + beta + r - 1/2 <= 0:
    raise ValueError('alpha + beta + r - 1/2 <= 0')
elif beta + r + 1/2 <= 0:
    raise ValueError('beta + r + 1/2 <= 0')
elif alpha + beta <= 0:
    raise ValueError('alpha + beta <= 0')
elif alpha - 1/2 <= 0:
    raise ValueError('alpha - 1/2 <= 0')

# EE
rate_ee = 1. - (1./(2+2*alpha+2*beta+2*r))

# FF
rate_ff = 1. if r>=0 else (1. - (-2*r/(1+2*alpha+2*beta)))

print("The EE rate is", rate_ee)
print("The FF rate is", rate_ff)
print("The", "EE" if rate_ee>=rate_ff else "FF", "estimator is more accurate")