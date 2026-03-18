class Data_container:
    def __init__(self,train_x,train_y, test_x,test_y,metadata=None):
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        self.metadata=metadata if metadata is not None else {}
        
    def get_train_data(self):
        return self.train_x,self.train_y
    def get_test_data(self):
        return self.test_x,self.test_y
    def get_shape(self):
        return {
            'train_x':self.train_x.shape if hasattr(self.train_x, 'shape') else len(self.train_x),
            'train_y':len(self.train_y),
            'test_x':self.test_x.shape if hasattr(self.test_x, 'shape') else len(self.test_x),
            'test_y':len(self.test_y)
        }
    def __repr__(self):
        shapes=self.get_shape()
        return (
            f"Data_container(train_x={shapes['train_x']}, train_y={shapes['train_y']},"
            f"test_sapmples={shapes['train_y_shape']}"
        )
    

class HierarchicalDataContainer:
    #used fro hierarchical data branching
    def __init__(self,parent_class,level,data_container,children=None):
        self.parent_class=parent_class
        self.level=level
        self.data=data_container
        self.children=children if children is not None else []
    def add_child(self,class_label,child_container):
        self.children[class_label] = child_container
    def get_child(self, class_label):
        return self.children.get(class_label)
    def get_all_children(self):
        return self.children
    def __repr__(self):
        return (
            f"HierarchicalDataContainer(parent_class={self.parent_class}',"
            f"level={self.level}, children={len(self.children)})"
        )
    
