from other_test import Shawn

class Monte( Shawn ):
    def __init__( self, c ):
        self.c = c 
    def other_func( self, d):
        return self.c + self.func( d ) 


instance = Shawn( a = 10 )
other_instance =  Monte( c = 5 ) 

print(other_instance.other_func( d = 4 )) 



