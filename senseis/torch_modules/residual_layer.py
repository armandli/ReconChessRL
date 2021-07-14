import torch.nn as nn

def downsampling1D(in_sz, out_sz, norm_layer):
  return nn.Sequential(
    nn.Linear(in_sz, out_sz),
    norm_layer(out_sz)
  )

def downsampling2DV1(in_c, out_c, ksz, norm_layer):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, bias=False),
    norm_layer(out_c)
  )

def downsampling2DV2(in_c, out_c, stride, norm_layer):
  return nn.Sequential(
    nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
    norm_layer(out_c),
  )

class ResidualLayer1DV1(nn.Module):
  def __init__(self, in_sz, out_sz, act_layer, norm_layer, downsample=None):
    super(ResidualLayer1DV1, self).__init__()
    self.fc1 = nn.Linear(in_sz, out_sz)
    self.fc2 = nn.Linear(out_sz, out_sz)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(out_sz)
    self.b2 = norm_layer(out_sz)
    self.downsample = downsample

  def forward(self, x):
    s = x
    x = self.fc1(x)
    x = self.b1(x)
    x = self.a1(x)
    x = self.fc2(x)
    x = self.b2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    x = self.a2(x)
    return x

class ResidualLayer1DV2(nn.Module):
  def __init__(self, in_sz, out_sz, act_layer, norm_layer, downsample=None):
    super(ResidualLayer1DV2, self).__init__()
    self.fc1 = nn.Linear(in_sz, out_sz)
    self.fc2 = nn.Linear(out_sz, out_sz)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_sz)
    self.b2 = norm_layer(out_sz)
    self.downsample = downsample

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.a1(x)
    x = self.fc1(x)
    x = self.b2(x)
    x = self.a2(x)
    x = self.fc2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x

# Add in dropout for 1D Residual Layer
class ResidualLayer1DV3(nn.Module):
  def __init__(self, in_sz, out_sz, act_layer, norm_layer, p, downsample=None):
    super(ResidualLayer1DV3, self).__init__()
    self.fc1 = nn.Linear(in_sz, out_sz)
    self.fc2 = nn.Linear(out_sz, out_sz)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_sz)
    self.b2 = norm_layer(out_sz)
    self.d1 = nn.Dropout(p=p)
    self.d2 = nn.Dropout(p=p)
    self.downsample = downsample

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.d1(x)
    x = self.a1(x)
    x = self.fc1(x)
    x = self.b2(x)
    x = self.d2(x)
    x = self.a2(x)
    x = self.fc2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x

# Automatic downsample based on in_sz != out_sz
class ResidualLayer1DV4(nn.Module):
  def __init__(self, in_sz, out_sz, act_layer, norm_layer, p):
    super(ResidualLayer1DV4, self).__init__()
    self.fc1 = nn.Linear(in_sz, out_sz)
    self.fc2 = nn.Linear(out_sz, out_sz)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_sz)
    self.b2 = norm_layer(out_sz)
    self.d1 = nn.Dropout(p=p)
    self.d2 = nn.Dropout(p=p)
    self.downsample = None
    if in_sz != out_sz:
      self.downsample = downsampling1D(in_sz, out_sz, norm_layer)

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.d1(x)
    x = self.a1(x)
    x = self.fc1(x)
    x = self.b2(x)
    x = self.d2(x)
    x = self.a2(x)
    x = self.fc2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x

# dropout is optional
class ResidualLayer1DV5(nn.Module):
  def __init__(self, in_sz, out_sz, act_layer, norm_layer, p=1.):
    super(ResidualLayer1DV5, self).__init__()
    self.p = p
    self.fc1 = nn.Linear(in_sz, out_sz)
    self.fc2 = nn.Linear(out_sz, out_sz)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_sz)
    self.b2 = norm_layer(out_sz)
    self.d1 = nn.Dropout(p=p)
    self.d2 = nn.Dropout(p=p)
    self.downsample = None
    if in_sz != out_sz:
      self.downsample = downsampling1D(in_sz, out_sz, norm_layer)

  def forward(self, x):
    s = x
    x = self.b1(x)
    if self.p < 1.:
      x = self.d1(x)
    x = self.a1(x)
    x = self.fc1(x)
    x = self.b2(x)
    if self.p < 1.:
      x = self.d2(x)
    x = self.a2(x)
    x = self.fc2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x

class ResidualLayer2DV1(nn.Module):
  def __init__(self, in_c, out_c, ksz, act_layer, norm_layer, downsample=None):
    super(ResidualLayer2DV1, self).__init__()
    self.c1 = nn.Conv2d(in_c, out_c, ksz, padding=int((ksz - 1) / 2), bias=False)
    self.c2 = nn.Conv2d(out_c, out_c, ksz, padding=int((ksz - 1) / 2), bias=False)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(out_c)
    self.b2 = norm_layer(out_c)
    self.downsample = downsample

  def forward(self, x):
    s = x
    x = self.c1(x)
    x = self.b1(x)
    x = self.a1(x)
    x = self.c2(x)
    x = self.b2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    x = self.a2(x)
    return x

class ResidualLayer2DV2(nn.Module):
  def __init__(self, in_c, out_c, ksz, act_layer, norm_layer, downsample=None):
    super(ResidualLayer2DV2, self).__init__()
    self.c1 = nn.Conv2d(in_c, out_c, ksz, padding=int((ksz - 1) / 2), bias=False)
    self.c2 = nn.Conv2d(out_c, out_c, ksz, padding=int((ksz - 1) / 2), bias=False)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_c)
    self.b2 = norm_layer(out_c)
    self.downsample = downsample

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.a1(x)
    x = self.c1(x)
    x = self.b2(x)
    x = self.a2(x)
    x = self.c2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x

# automatic downsample based on whether in_c != out_c and stride > 1
class ResidualLayer2DV3(nn.Module):
  def __init__(self, in_c, out_c, ksz, act_layer, norm_layer, stride=1):
    super(ResidualLayer2DV3, self).__init__()
    self.c1 = nn.Conv2d(in_c, out_c, ksz, stride=stride, padding=int((ksz - 1) / 2), bias=False)
    self.c2 = nn.Conv2d(out_c, out_c, ksz, padding=int((ksz - 1) / 2), bias=False)
    self.a1 = act_layer()
    self.a2 = act_layer()
    self.b1 = norm_layer(in_c)
    self.b2 = norm_layer(out_c)
    self.downsample = None
    if in_c != out_c or stride > 1:
      self.downsample = downsampling2DV2(in_c, out_c, stride, norm_layer)

  def forward(self, x):
    s = x
    x = self.b1(x)
    x = self.a1(x)
    x = self.c1(x)
    x = self.b2(x)
    x = self.a2(x)
    x = self.c2(x)
    if self.downsample is not None:
      s = self.downsample(s)
    x = x + s
    return x