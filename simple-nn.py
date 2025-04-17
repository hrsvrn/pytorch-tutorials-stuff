#Importing Libraries 
import torch  # Just the entire library
import torch.nn as nn  # All the neural network modules like nn.linear nn.conv1d etc etc loss functions are inside here
import torch.optim as optim  #All the optimization algos like stocastic gradient descent adam etc 
import torch.nn.functional as F #Activation fns like relu softmax 
from torch.utils.data import DataLoader #Dataloader gives easier data managment
import torchvision.datasets as datasets  #Pytorch has a lotta standard datasets .. we are going to use it to import mnist dataset for our project here
import torchvision.transforms as transforms #Transformations that we can perform on our dataset

#Create a fully connected network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,50)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

#Set Device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setting up our device 
#Hyperparameters
input_size=784 #28x28 will convert to 784 when unpacked
num_classes=10 # Classes are 0,1,2,3,4,5,6,7,8,9 i.e 10 classes
learning_rate=0.001 
batch_size=64 # number of training samples per iteration
num_epochs=10 # Number of times the model sees the whole dataset

#Load Data
train_dataset=datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True) # Downloading MNIST trained
test_dataset=datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True) # Downloading MNSIT tested
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) #DataLoading training set
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True) #Dataloading  testing set

#Initalize Network
model=NN(input_size=input_size,num_classes=num_classes).to(device) #Load model to device

#Loss and Optimizer
criterion=nn.CrossEntropyLoss() #Defining loss function here .. its Cross entropy loss
optimizer=optim.Adam(model.parameters(),lr=learning_rate)  #Defining Optimizer here... we are using Adam Optimizer


#Train Network
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader): #
        #Get data to cuda if possible
        data=data.to(device=device)
        targets=targets.to(device=device)

        data=data.reshape(data.shape[0],-1)# Reshaping 28x28 to 724

        #forward
        scores=model(data) # Get the training tensors back
        loss=criterion(scores,targets)  # Calculating loss


        #backward
        optimizer.zero_grad() # Clear old mistakes or loss in general
        loss.backward() #Analyze mistake

        #Gradient Descent  or adam step
        optimizer.step() #Update knowledge

#Check accuracy on training and testing to see how good our model is 

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuract on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)  # Fixed reshaping
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()  # Added .item()
            num_samples += predictions.size(0)
    model.train()
    acc = num_correct / num_samples if num_samples > 0 else 0  # Division guard
    print(f"Got {num_correct}/{num_samples} with accuracy {acc*100:.2f}%")

# During training loop
print("Checking accuracy on TRAINING data")
check_accuracy(train_loader, model)  # Use training loader

print("\nChecking accuracy on TEST data")
check_accuracy(test_loader, model)  # Use test loader

