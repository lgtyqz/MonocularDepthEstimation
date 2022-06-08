# Model setup (!!)
# Two models: One for depth, one for color constancy
# Then post-process and visualize accordingly
def convSize(orig, kernelSize, stride):
  return math.floor((orig - (kernelSize - 1) - 1)/stride + 1)

# This one takes in lists for kernelSize and stride
def repeatedConvSize(orig, kernelSize, stride):
  assert len(kernelSize) == len(stride), "bro the lists kernelSize and stride must be equal length"
  size = orig
  for i, k in enumerate(kernelSize):
    size = convSize(size, kernelSize[i], stride[i])
  return size

class DepthEstimator(nn.Module):
  """
  initialization notes:
  image_size is a tuple of the image size (W, H) (lol)
  """
  def __init__(self, image_size):
    # Image size of the nyu images is like 640x480, with 1449 examples
    # Might make the math for the linear layer a bit tricky
    super(DepthEstimator, self).__init__()
    self.image_size = image_size
    # Part 1: convolutional setup
    self.conv1 = nn.Conv2d(3, 6, 5) # Each image channel (R, G, B) is a separate layer
    self.conv2 = nn.Conv2d(6, 16, 5)
    # self.conv3 = nn.Conv2d(16, 32, 3)
    self.maxP1 = nn.MaxPool2d(4, 4, return_indices=True)
    self.maxP2 = nn.MaxPool2d(4, 4, return_indices=True)
    # self.maxP3 = nn.MaxPool2d(2, 2, return_indices=True)
    # self.linear_size = (
    #   repeatedConvSize(self.image_size[0], [5, 4, 5, 2, 3, 2], [1, 4, 1, 2, 1, 2]),
    #   repeatedConvSize(self.image_size[1], [5, 4, 5, 2, 3, 2], [1, 4, 1, 2, 1, 2])
    # )
    self.linear_size = (
      repeatedConvSize(self.image_size[0], [5, 4, 5, 4], [1, 4, 1, 4]),
      repeatedConvSize(self.image_size[1], [5, 4, 5, 4], [1, 4, 1, 4])
    )
    print(self.linear_size)
    self.fcIn = nn.Linear(16 * self.linear_size[0] * self.linear_size[1], 8 * self.linear_size[0] * self.linear_size[1]) # TODO: Make the numbers here
    # Part 2: And now starts the deconvolutional thingos
    self.fcOut = nn.Linear(8 * self.linear_size[0] * self.linear_size[1], 16 * self.linear_size[0] * self.linear_size[1])
    self.deconv1 = nn.ConvTranspose2d(16, 6, 5)
    self.deconv2 = nn.ConvTranspose2d(6, 1, 5)
    # self.deconv3 = nn.ConvTranspose2d(6, 1, 5)
    # self.unpool1 = nn.MaxUnpool2d(2, 2)
    self.unpool2 = nn.MaxUnpool2d(4, 4)
    self.unpool3 = nn.MaxUnpool2d(4, 4)
    self.relu = nn.ReLU()

  def forward(self, x):
    # print(type(x))
    x = self.conv1(x)
    self.preMaxP1 = x.size()
    x, ind1 = self.maxP1(x)
    # print(ind1.dtype)
    # print(torch.max(ind1), torch.min(ind1))

    x = self.conv2(x)
    self.preMaxP2 = x.size()
    x, ind2 = self.maxP2(x)

    # x = self.conv3(x)
    # self.preMaxP3 = x.size()
    # x, ind3 = self.maxP3(x)
    # print(ind1.shape, ind2.shape, ind3.shape)
    ogShape = tuple(x.shape)
    # print(ogShape)
    x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
    x = self.relu(self.fcIn(x))
    x = self.relu(self.fcOut(x))
    x = torch.reshape(x, ogShape)
    # print(x.dtype)
    x = self.deconv1(self.unpool2(x, ind2, output_size=self.preMaxP2))
    x = self.deconv2(self.unpool3(x, ind1, output_size=self.preMaxP1))
    # x = self.deconv3(self.unpool3(x, ind1, output_size=self.preMaxP1))
    return x