# ����Mapreduce��KNNʵ��

## ��Ŀ����

- ����Ŀʵ����KNN�㷨��Hadoopƽ̨����***ŷ������***��***��Ȩŷ������***��***��˹����***��MapReduceʵ�֡�
- ��ɫ���⣺������KNNʵ�ֵ������������˻���***ŷ������***��***��Ȩŷ������***��***��˹����***��ʵ�֡�

- �������������[http://archive.ics.uci.edu/ml/datasets/Iris](http://archive.ics.uci.edu/ml/datasets/Iris)��

  ʹ�õ����������β�����ݼ����ݼ��ڰ��� 3 �๲ 150 ����¼��ÿ��� 50 �����ݣ�ÿ����¼���� 4 �����������೤�ȡ�������ȡ����곤�ȡ�������ȣ�����ͨ����4������Ԥ���β�������ڣ�iris-setosa, iris-versicolour, iris-virginica���е���һƷ�֡�

  ѵ����������Ϊ������ֵ1������ֵ2��.......����ǩ

  ���Լ�������Ϊ������ֵ1������ֵ2��.......,   ��ȷ��ǩ

## ���з���

- ���г���ʱ��Ҫ���� **hadoop jar KNN.jar KNN train/iris_train.csv output**
  - ִ����ͨKNN��Ҫ�Ѽ�ȨKNN�͸�˹����ע�͵�
  - ִ�м�ȨKNN���KNN�͸�˹����ע�͵�
  - Reduce��ҲҪ��Ӧ��ע�͵�

## �����ļ�

����hadoopֻ�谴��ʵ����ٹ�һ������ˣ��˴���ֱ�Ӳ��ó���ϰ�õĻ�������ʵ�顣

## ��������

### mapper

mapper�������Ƕ�ȡ���Լ����������������ѵ�������������ƶȡ�

- setup���𽫲��Լ����ݶ��������ÿһ�д�Ϊһ��list���ٽ���Щlist����һ�����Լ�list�У���ArrayList<ArrayList<Float>>��
  ```java
  protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
              // load the test vectors
              FileSystem fs = FileSystem.get(context.getConfiguration());
              BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                      "org.niubility.learning.test", "./test/iris_test_data.csv")))));//��ȡ��·���µĲ��Լ�
              String line = br.readLine();
              int count = 0;
              while (line != null) {//���ж�ȡ
                  String[] s = line.split(",");//��,�ָ�
                  ArrayList<Float> testcase = new ArrayList<Float>();
                  for (int i = 0; i < s.length-1; i++){
                      testcase.add(Float.parseFloat(s[i]));//�洢��list��
                  }
                  test.add(testcase);//���洢ÿһ�е�list�ŵ���һ��list��
                  line = br.readLine();
                  count++;
              }
              br.close();
          }
  ```

- map����������������ѵ������������

  ```java
  public void map(Object key, Text value, Context context) throws IOException, InterruptedException  
          {  
              //key��ѵ�������к�
              context.setStatus(key.toString()); //����
              String[] s = value.toString().split(",");//��value����,���ֿ�
              String label = s[s.length - 1]; //��ǩ�洢���е����һ��
              for (int i=0; i<test.size(); i++){//����test
                  ArrayList<Float> curr_test = test.get(i);//��ȡÿһ�е�list
                  double tmp = 0;
                  for(int j=0; j<curr_test.size(); j++){
                      //��������ѵ�������� x^2+y^2+...
                      tmp += (curr_test.get(j) - Float.parseFloat(s[j]))*(curr_test.get(j) - Float.parseFloat(s[j]));
                  }
                  context.write(new Text(Integer.toString(i)), new Text(Double.toString(tmp)+","+label)); //�����������,����ѵ��������&��ǩ                 
              }
          } 
  ```

### combiner

combiner���趨kֵ��������ѡ��ǰk������ȡ����������Ϊ�����Ȼ��ֱ����ŷ�����룬��Ȩŷ�����룬��˹�����õ������

```java
public static class KNNCombiner extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            ArrayList<Dis_Label> dis_Label_set = new ArrayList<Dis_Label>();
            for (Text value : values){
                String[] s = value.toString().split(","); //������ ����+��ǩ
                Dis_Label tmp = new Dis_Label();
                tmp.label = s[1];
                tmp.dis = Float.parseFloat(s[0]);//תΪfloat
                dis_Label_set.add(tmp);//����list��
            }
            //���� 
            Collections.sort(dis_Label_set, new Comparator<Dis_Label>(){
                @Override
                public int compare(Dis_Label a, Dis_Label b){ 
                    if (a.dis > b.dis){
                        return 1; //С����ǰ
                    }
                    return -1;
                }
            });

            final int k = 5; //Kֵ
			����������������
        }  
    }  
```

�������ּ��㷽����Ӧ�Ĳ������£�

```java
//ͳ��ǰK����������ı�ǩ KNN
for (int i=0; i<dis_Label_set.size() && i<k; i++){
    context.write(key, new Text(dis_Label_set.get(i).label));
}
//KNN end
```

```java
//��ȨKNN
HashMap<String, Double> label_dis = new HashMap<String, Double>(); //label , confidence
for (int i=0; i<dis_Label_set.size() && i<k; i++){
     String cur_l = dis_Label_set.get(i).label;
     if (!label_dis.containsKey(cur_l)) label_dis.put(cur_l, 0.0);
     label_dis.put(cur_l, label_dis.get(cur_l) + 1.0/(0.5+dis_Label_set.get(i).dis));
}
for (String l:label_dis.keySet()){
      context.write(key, new Text(Double.toString(label_dis.get(l))+","+l));
             }
//��ȨKNN End
```

```java
//��˹����
HashMap<String, Double> label_dis = new HashMap<String, Double>(); //label , confidence
for (int i=0; i<dis_Label_set.size() && i<k; i++){
    String cur_l = dis_Label_set.get(i).label;
    if (!label_dis.containsKey(cur_l)) label_dis.put(cur_l, 0.0);
    label_dis.put(cur_l, label_dis.get(cur_l) + Math.exp(Math.pow(dis_Label_set.get(i).dis,2)/0.5));
}
for (String l:label_dis.keySet()){
    context.write(key, new Text(Double.toString(label_dis.get(l))+","+l));
}
//��˹���� End
```

ʵ������ʱ����Ҫע�͵��������ּ��㡣

### reducer

reducer���ǽ���������������ȡ���Ӧ�ı�ǩ���������

```java
public static class KNNReducer extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            //KNN����
            // HashMap<String, Integer> ans = new HashMap<String, Integer>();
            // for(Text val:values)  
            // {  
            //     if (!ans.containsKey(val)){
            //         ans.put(val.toString(), 0);
            //     }
            //     ans.put(val.toString(), ans.get(val.toString())+1); 
            // }  
            // //ȷ����ǩ
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

            //��ȨKNN & ��˹����
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
            //��ȨKNN & ��˹ End

        }
    }  
```

ͬ���أ�ʵ������ʱ��Ҫע�͵��������֡�

## �ļ����

```
Hadoop_knn
������ img  //���н����ͼ
������ result  //��Ž������ļ���
|    ������ part-r-00000(1)  //����ŷ�����������
|    ������ part-r-00000(2)  //���ڼ�Ȩŷ�����������
|    ������ part-r-00000(3)  //���ڸ�˹����������
������ test //���Լ��ļ���
|    ������ iris_test_data.csv
����
������ train //���Լ��ļ���
|    ������ iris_train.csv
������ KNN.jar  
������ KNN.java  //Դ�����ļ�
������ README.MD  
������ REPORT.MD //�����ļ�
```


