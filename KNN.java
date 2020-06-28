import java.io.IOException;  
import java.util.StringTokenizer;  
import org.apache.hadoop.conf.Configuration;  
import org.apache.hadoop.fs.Path;  
import org.apache.hadoop.io.Text;  
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;  
import org.apache.hadoop.mapreduce.Mapper;  
import org.apache.hadoop.mapreduce.Reducer;  
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;  
import org.apache.hadoop.mapreduce.lib.input.FileSplit;  
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;  
import org.apache.hadoop.util.GenericOptionsParser;  
  
import java.util.Collections;
import java.util.Comparator;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

public class KNN  
{  
      
    public static class Dis_Label {
        public float dis;//距离
        public String label;//标签
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {  
        private ArrayList<ArrayList<Float>> test = new ArrayList<ArrayList<Float>> ();

        @Override  
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException  
        {  
            //key是训练数据行号
            context.setStatus(key.toString());
            String[] s = value.toString().split(",");
            String label = s[s.length - 1];        
            for (int i=0; i<test.size(); i++){
                ArrayList<Float> curr_test = test.get(i);
                double tmp = 0;
                for(int j=0; j<curr_test.size(); j++){
                    tmp += (curr_test.get(j) - Float.parseFloat(s[j]))*(curr_test.get(j) - Float.parseFloat(s[j]));
                }
                context.write(new Text(Integer.toString(i)), new Text(Double.toString(tmp)+","+label)); //测试样例编号,所有训练集距离&标签                 
            }

        }  
        protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
            // load the test vectors
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                    "org.niubility.learning.test", "./test/iris_test_data.csv")))));
            String line = br.readLine();
            int count = 0;
            while (line != null) {
                String[] s = line.split(",");
                ArrayList<Float> testcase = new ArrayList<Float>();
                for (int i = 0; i < s.length-1; i++){
                    testcase.add(Float.parseFloat(s[i]));
                }
                test.add(testcase);
                line = br.readLine();
                count++;
            }
            br.close();
        }
    }  

    public static class KNNCombiner extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            ArrayList<Dis_Label> dis_Label_set = new ArrayList<Dis_Label>();
            for (Text value : values){
                String[] s = value.toString().split(","); //拆开所有 距离+标签
                Dis_Label tmp = new Dis_Label();
                tmp.label = s[1];
                tmp.dis = Float.parseFloat(s[0]);
                dis_Label_set.add(tmp);
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

            /****
            *   执行普通KNN请把加权KNN和高斯函数注释掉
            *   执行加权KNN请把KNN和高斯函数注释掉
            *   Reduce里也要相应地注释掉
            ***/

            //统计前K个最近样例的标签 KNN
            //for (int i=0; i<dis_Label_set.size() && i<k; i++){
            //    context.write(key, new Text(dis_Label_set.get(i).label));
            //}
            //KNN end

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

            //高斯函数
            //HashMap<String, Double> label_dis = new HashMap<String, Double>(); //label , confidence
            //for (int i=0; i<dis_Label_set.size() && i<k; i++){
            //    String cur_l = dis_Label_set.get(i).label;
            //    if (!label_dis.containsKey(cur_l)) label_dis.put(cur_l, 0.0);
            //    label_dis.put(cur_l, label_dis.get(cur_l) + Math.exp(Math.pow(dis_Label_set.get(i).dis,2)/0.5));
            //}
            //for (String l:label_dis.keySet()){
            //    context.write(key, new Text(Double.toString(label_dis.get(l))+","+l));
            //}
            //高斯函数 End
        }  
    }  

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
      
    public static void main(String[] args) throws Exception  
    {  
        Configuration conf=new Configuration();  
        //String[] otherArgs=new GenericOptionsParser(conf, args).getRemainingArgs();  
        if(args.length!=2)  
        {  
            System.out.println("Usage: InvertedIndex <in> <out>");  
            System.exit(2);  
        }  
        Path inputPath=new Path(args[0]);  
        Path outputPath=new Path(args[1]);  
        outputPath.getFileSystem(conf).delete(outputPath, true);  
          
        //The constructor Job(Configuration conf, String jobName) is deprecated  
        //Job job=new Job(conf, "Inverted Index");  
        Job job=Job.getInstance(conf, "KNN");  
        job.setJarByClass(KNN.class);  
          
        job.setMapperClass(TokenizerMapper.class);  
        job.setMapOutputKeyClass(Text.class);  
        job.setMapOutputValueClass(Text.class);  
           
        job.setCombinerClass(KNNCombiner.class);  
          
        job.setReducerClass(KNNReducer.class);          
        job.setOutputKeyClass(Text.class);  
        job.setOutputValueClass(Text.class);  
          
        FileInputFormat.addInputPath(job, inputPath);  
        FileOutputFormat.setOutputPath(job, outputPath);  
          
        System.exit(job.waitForCompletion(true)? 0:1);  
    }  
      
}  
