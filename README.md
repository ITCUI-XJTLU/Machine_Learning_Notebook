# HongyiLee_MLcourse
This a collection of homwork I do for the Hongyi_Lee's online courses on machine learning and deep learning. Now, I have finished the first homework , I will continue to complete the rest of homework in my spare time in university . So, updating soon ...  
## Homework 1: Regression  
### (1) Work 1: 
The homework is aimed to use the data of first nine day PM2.5 information to predicted the MP2.5 on the 10th day. Instead of simply using the functions encapsulated in some libraries, I programmed the codes to make a regression by myself . Therefore, the codes can only be used to the models ,like y=ax+bX^2, and it can not be applied to the more complex models.  
There are the main steps of the project:
* Preproccession: clean and preproccess the origin data to the one we need 
* Nomalization 
* Train : build my model and use Adagrad to opitimize the model  
* Test : use the test data to see how well the model to make a prediction   
* Prediction 
>If you want to see my codes , please turn to the [master branch](https://github.com/ITCUI-XJTLU/HongyiLee_MLcourse/tree/master/H1)  
  
  
### (2) Work 2:  
Since I have learned many machine learning opitimiters on the course , I tend to see and compare the performance of the four popular optimizers: `Adagray` , `RMSprop` , `SDGm` , `Adam` .   
Please view [Adarag](https://www.youtube.com/watch?list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=705&v=yKKNr-QKz2Q&feature=youtu.be) , [RMSprop](https://www.youtube.com/watch?v=5Yt-obwvMHI) , [Adam](https://www.youtube.com/watch?v=JXQT_vxqwIs) , if you want to learn more about the optimizers  
Please view the [optimizer.py](https://github.com/ITCUI-XJTLU/HongyiLee_MLcourse/blob/master/H1/optimizer.py) , if you want to see the detail of how I program these four optimizers.   
And the result is :  
<div align=center><img width="550" height="430" src="https://github.com/ITCUI-XJTLU/HongyiLee_MLcourse/raw/master/H1/Four_Optimizers.png"/></div>  
  
From the graph above , we know that `SDGm` perform the best . And in most of other machine learning models, the Adam is actually the most commen and robust method we use to optimize models .
