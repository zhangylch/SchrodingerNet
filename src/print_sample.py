import time

class Print_Info():
    def __init__(self, ferr):
        self.ferr = ferr
   
    def __call__(self, iepoch, lr, eig_train, loss_train, loss_val):
        self.forward(iepoch, lr, eig_train, loss_train, loss_val)

    def forward(self, iepoch, lr, eig_train, loss_train, loss_val):
        #output the error 
        self.ferr.write("Step= {:6},  lr= {:5e}  ".format(iepoch, lr))
        self.ferr.write("eigene: ")
        self.ferr.write("{:10e} ".format(eig_train))
        self.ferr.write(" loss: ")
        self.ferr.write("{:10e} / {:10e}".format(loss_train, loss_val))
        self.ferr.write(" \n")
        self.ferr.flush()
        
