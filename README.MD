# 基于Mapreduce的KNN实现

## 项目介绍

- 该项目实现了KNN算法在Hadoop平台基于***欧拉距离***，***加权欧拉距离***，***高斯函数***的MapReduce实现。
- 特色或创意：在网上KNN实现的例子上添加了基于***欧拉距离***，***加权欧拉距离***，***高斯函数***的实现。

- 解决的问题来自[http://archive.ics.uci.edu/ml/datasets/Iris](http://archive.ics.uci.edu/ml/datasets/Iris)。

  使用的是著名的鸢尾花数据集。据集内包含 3 类共 150 条记录，每类各 50 个数据，每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。

  训练集中数据为：属性值1，属性值2，.......，标签

  测试集中数据为：属性值1，属性值2，.......,   正确标签

## 运行方法

- 运行程序时需要输入 **hadoop jar KNN.jar KNN train/iris_train.csv output**
  - 执行普通KNN需要把加权KNN和高斯函数注释掉
  - 执行加权KNN请把KNN和高斯函数注释掉
  - Reduce里也要相应地注释掉

## 部署文件

部署hadoop只需按照实验二再过一遍就行了，此处我直接采用超算习堂的环境进行实验。

## 代码描述

### mapper

mapper的任务是读取测试集并计算测试样本与训练集样本的相似度。

- setup负责将测试集数据读入进来，每一行存为一个list，再将这些list存入一个测试集list中（即ArrayList<ArrayList<Float>>）
  ```java
  protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
              // load the test vectors
              FileSystem fs = FileSystem.get(context.getConfiguration());
              BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                      "org.niubility.learning.test", "./test/iris_test_data.csv")))));//读取该路径下的测试集
              String line = br.readLine();
              int count = 0;
              while (line != null) {//按行读取
                  String[] s = line.split(",");//以,分割
                  ArrayList<Float> testcase = new ArrayList<Float>();
                  for (int i = 0; i < s.length-1; i++){
                      testcase.add(Float.parseFloat(s[i]));//存储到list中
                  }
                  test.add(testcase);//将存储每一行的list放到另一个list中
                  line = br.readLine();
                  count++;
              }
              br.close();
          }
  ```

- map负责计算测试样本与训练集样本距离

  ```java
  public void map(Object key, Text value, Context context) throws IOException, InterruptedException  
          {  
              //key是训练数据行号
              context.setStatus(key.toString()); //设置
              String[] s = value.toString().split(",");//将value按‘,’分开
              String label = s[s.length - 1]; //标签存储在行的最后一个
              for (int i=0; i<test.size(); i++){//遍历test
                  ArrayList<Float> curr_test = test.get(i);//读取每一行的list
                  double tmp = 0;
                  for(int j=0; j<curr_test.size(); j++){
                      //计算所有训练集距离 x^2+y^2+...
                      tmp += (curr_test.get(j) - Float.parseFloat(s[j]))*(curr_test.get(j) - Float.parseFloat(s[j]));
                  }
                  context.write(new Text(Integer.toString(i)), new Text(Double.toString(tmp)+","+label)); //测试样例编号,所有训练集距离&标签                 
              }
          } 
  ```

### combiner

combiner中设定k值，即排序选择前k个，并取其中最多的作为结果。然后分别根据欧拉距离，加权欧拉距离，高斯函数得到结果。

```java
public static class KNNCombiner extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            ArrayList<Dis_Label> dis_Label_set = new ArrayList<Dis_Label>();
            for (Text value : values){
                String[] s = value.toString().split(","); //拆开所有 距离+标签
                Dis_Label tmp = new Dis_Label();
                tmp.label = s[1];
                tmp.dis = Float.parseFloat(s[0]);//转为float
                dis_Label_set.add(tmp);//加入list中
            }
            //排序 
            Collections.sort(dis_Label_set, new Comparator<Dis_Label>(){
                @Override
                public int compare(Dis_Label a, Dis_Label b){ 
                    if (a.dis > b.dis){
                        return 1; //小的在前
                    }
                    return -1;
                }
            });

            final int k = 5; //K值
			········
        }  
    }  
```

其中三种计算方法对应的操作如下：

```java
//统计前K个最近样例的标签 KNN
for (int i=0; i<dis_Label_set.size() && i<k; i++){
    context.write(key, new Text(dis_Label_set.get(i).label));
}
//KNN end
```

```java
//加权KNN
HashMap<String, Double> label_dis = new HashMap<String, Double>(); //label , confidence
for (int i=0; i<dis_Label_set.size() && i<k; i++){
     String cur_l = dis_Label_set.get(i).label;
     if (!label_dis.containsKey(cur_l)) label_dis.put(cur_l, 0.0);
     label_dis.put(cur_l, label_dis.get(cur_l) + 1.0/(0.5+dis_Label_set.get(i).dis));
}
for (String l:label_dis.keySet()){
      context.write(key, new Text(Double.toString(label_dis.get(l))+","+l));
             }
//加权KNN End
```

```java
//高斯函数
HashMap<String, Double> label_dis = new HashMap<String, Double>(); //label , confidence
for (int i=0; i<dis_Label_set.size() && i<k; i++){
    String cur_l = dis_Label_set.get(i).label;
    if (!label_dis.containsKey(cur_l)) label_dis.put(cur_l, 0.0);
    label_dis.put(cur_l, label_dis.get(cur_l) + Math.exp(Math.pow(dis_Label_set.get(i).dis,2)/0.5));
}
for (String l:label_dis.keySet()){
    context.write(key, new Text(Double.toString(label_dis.get(l))+","+l));
}
//高斯函数 End
```

实际运行时，需要注释掉其他两种计算。

### reducer

reducer就是将结果进行输出，获取其对应的标签输出出来。

```java
public static class KNNReducer extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            //KNN部分
            // HashMap<String, Integer> ans = new HashMap<String, Integer>();
            // for(Text val:values)  
            // {  
            //     if (!ans.containsKey(val)){
            //         ans.put(val.toString(), 0);
            //     }
            //     ans.put(val.toString(), ans.get(val.toString())+1); 
            // }  
            // //确定标签
            // int mx = -1;
            // String ansLabel = "";
            // for (String l:ans.keySet()){
            //     if (mx < ans.get(l)){
            //         mx = ans.get(l);
            //         ansLabel = l;
            //     }
            // }   
            // context.write(key, new Text(ansLabel));  
            //KNN End

            //加权KNN & 高斯函数
            double mx = -1;
            String ansLabel = "";
            for (Text val: values){
                String[] s = val.toString().split(","); 
                if (Double.parseDouble(s[0]) > mx){
                    mx = Double.parseDouble(s[0]);
                    ansLabel = s[1];
                }
            }
            context.write(key, new Text(ansLabel));
            //加权KNN & 高斯 End

        }
    }  
```

同样地，实际运行时需要注释掉其余两种。

## 文件组成

```
Hadoop_knn
├── img  //运行结果截图
├── result  //存放结果输出文件夹
|    ├── part-r-00000(1)  //基于欧拉距离计算结果
|    ├── part-r-00000(2)  //基于加权欧拉距离计算结果
|    ├── part-r-00000(3)  //基于高斯函数计算结果
├── test //测试集文件夹
|    ├── iris_test_data.csv
类型
├── train //测试集文件夹
|    ├── iris_train.csv
├── KNN.jar  
├── KNN.java  //源代码文件
├── README.MD  
├── REPORT.MD //报告文件
```



