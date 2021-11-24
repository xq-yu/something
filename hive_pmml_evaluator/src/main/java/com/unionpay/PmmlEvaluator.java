package com.unionpay;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.jpmml.evaluator.Evaluator;
import org.dmg.pmml.FieldName;
import org.eclipse.jetty.util.ajax.JSON;
import org.jpmml.evaluator.FieldValue;
import java.io.*;
import java.util.*;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.TargetField;
import org.xml.sax.SAXException;
import java.util.LinkedHashMap;
import javax.xml.bind.JAXBException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import org.jpmml.evaluator.HasProbability;

public class PmmlEvaluator extends UDF {
    static ObjectMapper objectMapper = new ObjectMapper();

    // pmml模型预测
    public String evaluate(String job, String input)
            throws JsonProcessingException, IOException, SAXException, JAXBException {
        if(job.equals("loadmodel")){
            PmmlUtil.loadModel(input);
            return "model "+input+" has been loaded";
        }
        else{
            //创建预测模型
            Evaluator model = PmmlUtil.getEvaluator(); 
            List<? extends InputField> inputFields = model.getInputFields();
            List<? extends TargetField> targetFields = model.getTargetFields();
            Iterator var11 = inputFields.iterator();

            // 解析输入json
            String js=input;
            JsonNode jsonNode = objectMapper.readTree(js);
            Map<FieldName, FieldValue> arguments = new LinkedHashMap();
            while (var11.hasNext()) {
                InputField inputField = (InputField) var11.next();
                FieldName inputName = inputField.getName();
                String name = inputName.getValue();

                String value = jsonNode.get(name).asText();
                FieldValue inputValue = inputField.prepare(value);
                arguments.put(inputName, inputValue);
            }

            // pmml模型预测(默认一个结果)
            Map<FieldName, ?> results = model.evaluate(arguments);
            Iterator it = results.keySet().iterator();
            Object targetValue = results.get(it.next());

            // 对于存在概率的结果
            Map<String, Double> result = new LinkedHashMap();
            if (targetValue instanceof HasProbability) {
                HasProbability hasProbability = (HasProbability) targetValue;
                Set<?> categories = hasProbability.getCategories();
                for (Object category : categories) {
                    Double categoryProbability = hasProbability.getProbability(category);
                    result.put(category.toString(), categoryProbability);
                }
            }
            return JSON.toString(result);
        }

    }

    public static void main(String[] args) throws JsonProcessingException, IOException, SAXException, JAXBException {
        PmmlEvaluator evaluator = new PmmlEvaluator();
        String features = "{\"Sepal_Length\":\"6\", \"Sepal_Width\":\"4\", \"Petal_Length\":\"3\", \"Petal_Width\":\"2\"}";
        features = "{\"credit_charitable_days_3m\":-0.0341,\"debit_fail_amt_1m\":0.0004,\"debit_bill_amt_3m\":-0.1137,\"debit_pos_amt_6m\":0.1635,\"debit_bill_days_3m\":-0.0609,\"act_scan_cnt_6m\":-0.0078,\"debit_online_amt_12m\":0.2938,\"birth_pro_encoding\":-0.4373,\"view_before_dawn_days_3m\":-0.0642,\"diff_regdt_enter_days\":-0.7042,\"debit_convenient_amt_12m\":0.0897,\"debit_lack_fund_days_12m\":-0.0321,\"pass_scan_amt_3m\":0.145,\"debit_max_single_amt_1m\":0.3104,\"debit_lack_fund_days_3m\":-0.1472,\"credit_public_days_12m\":0.0594,\"rmd_cnt_3m\":0.03,\"act_scan_cnt_3m\":-0.024,\"debit_public_amt_12m\":0.064,\"view_midnight_days_1m\":-0.0635,\"multi_xd_days_3m\":0.0313,\"mean_coupon_amt_6m\":0.171,\"gps_city_per_3m\":-0.1392,\"age\":0.6197,\"max_bill_total_at\":0.1004}";
        String pmmlName = "/Users/yu/Desktop/coding/java/hive_pmml_evaluator/src/main/java/com/unionpay/DecisionTreeIris.pmml";
        pmmlName="/Users/yu/Desktop/coding/java/hive_pmml_evaluator/src/main/java/com/unionpay/model_v1.pmml";
        System.out.println(evaluator.evaluate(features, pmmlName));
    }
}