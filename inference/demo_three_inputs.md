# ID=1 Fact Checking任务的三类真实API输入示例

## 查询信息
- **ID**: 1
- **文件名**: employment-table01
- **问题**: Match the year where the employment in agriculture was the highest and state its corresponding total employed population in the year.
- **问题类型**: Fact Checking > Multi-hop Fact Checking
- **期望答案**: 1955, 62170
- **答案格式**: Year, Population

## COT字段说明
数据集中的COT字段内容（人工标注）：
```json
"COT": [
  {"planning": "First, we need to clean the data and extract relevant columns."},
  {"planning": "Next, we identify the year with the maximum employment in agriculture."},
  {"planning": "Then, we match this year to its corresponding total employed population."},
  {"planning": "Finally, we extract and format the final answer."}
]
```

**重要**：此COT字段**不会发送给API**，仅用于数据集分析和验证。Prompt中的"Chain-of-thought"是要求模型自己推理。

---

## 输入类型1：Image-only 模态

### API调用格式（OpenAI格式）
```json
[{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "<下方完整提示词>"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
      }
    }
  ]
}]
```

### 完整提示词内容
```text
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.

# Chain-of-thought 
Let's think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.

# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. If the question is judgmental, please answer 'Yes' or 'No'.
I will give you a table image and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...

Let's get start!
# Table
[See the table image]
# Question
Match the year where the employment in agriculture was the highest and state its corresponding total employed population in the year.
Emphasize: [Final Answer]: Year, Population
```

**特点**：
- 纯视觉输入，表格以PNG图片形式提供
- 提示词中Table部分为占位符"[See the table image]"
- 图片采用base64编码嵌入API请求

---

## 输入类型2：Text-only 模态（HTML格式）

### API调用格式（OpenAI格式）
```json
[{
  "role": "user",
  "content": "<下方完整提示词（含HTML表格）>"
}]
```

### 完整提示词内容
```text
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.

# Chain-of-thought
Let's think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.

# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. If the question is judgmental, please answer 'Yes' or 'No'.
I will give you a table and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...

Let's get start!
# Table
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th colspan="2" halign="left">Civilian labor force</th>
      <th colspan="2" halign="left">Total employed</th>
      <th colspan="2" halign="left">Agriculture</th>
      <th colspan="2" halign="left">Nonagriculture</th>
      <th>Unemployed</th>
      <th>Unemployment rate</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Total</th>
      <th>Percent</th>
      <th>Number</th>
      <th>Percent</th>
      <th>Number</th>
      <th>Percent</th>
      <th>Number</th>
      <th>Percent</th>
      <th>Number</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1947</td>
      <td>60168</td>
      <td>100.0</td>
      <td>57812</td>
      <td>100.0</td>
      <td>8256</td>
      <td>14.3</td>
      <td>49556</td>
      <td>85.7</td>
      <td>2356</td>
      <td>3.9</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1948</td>
      <td>61442</td>
      <td>100.0</td>
      <td>59357</td>
      <td>100.0</td>
      <td>7629</td>
      <td>12.9</td>
      <td>51728</td>
      <td>87.1</td>
      <td>2064</td>
      <td>3.4</td>
    </tr>
    ... [更多行] ...
  </tbody>
</table>

# Question
Match the year where the employment in agriculture was the highest and state its corresponding total employed population in the year.
Emphasize: [Final Answer]: Year, Population
```

**特点**：
- 纯文本输入，不使用图片
- 表格以HTML格式完整嵌入提示词
- 提示词+HTML总长度约51KB（本例）
- 可选格式：HTML/CSV/Markdown/LaTeX

---

## 输入类型3：Mix 模态（Image + HTML）

### API调用格式（OpenAI格式）
```json
[{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "<下方完整提示词（含HTML表格）>"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
      }
    }
  ]
}]
```

### 完整提示词内容
```text
# Role Play
Suppose you are an expert in table analysis and your task is to provide answers to questions based on the content of the table.

# Chain-of-thought 
Let's think step by step as follows and make the most of your strengths as a table analysis expert：
1、Fully understand the question and extract the necessary information from it.
2、Clearly and comprehensively understanding the content of  the table, including the structure of the table, the meaning and formatting of each row and column header（Note: There is usually summative cell in the table, such as all, combine, total, sum, average, mean, etc. Please pay careful attention to the flag information in the row header and column header, this information can help you to skip many operations.）
3、Based on the question, select the row and column headers in the table that are most relevant to it and find the corresponding cells based on them.
4、According to the requirements of the question, perform statistical, calculation, ranking, or other operations on the cells you selected, and output of the answer in the format specified by the definition.

# Output Control
1、The final answer should be one-line and strictly follow the format: "[Final Answer]: AnswerName1, AnswerName2...". 
2、Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. Give the final answer to the question directly without any explanation. If the question is judgmental, please answer 'Yes' or 'No'.
I will give you a table and a question, please use the table to answer the question in specified format: [Final Answer]: AnswerName1, AnswerName2...

Let's get start!
# Table
<table border="1" class="dataframe">
  ... [完整HTML表格，与Text-only相同] ...
</table>

# Question
Match the year where the employment in agriculture was the highest and state its corresponding total employed population in the year.
Emphasize: [Final Answer]: Year, Population
```

**特点**：
- 多模态输入：HTML文本 + PNG图片
- 提示词与Text-only完全相同（包含完整HTML表格）
- 额外提供PNG图片的base64编码
- 测试模型能否融合文本和视觉信息

---

## 三类输入对比总结

| 模态 | 文本内容 | 图片 | 表格信息来源 | 提示词长度 |
|------|---------|------|-------------|-----------|
| Image-only | 任务指令 + "[See the table image]" | ✅ PNG | 仅视觉 | ~1.2KB |
| Text-only | 任务指令 + 完整HTML表格 | ❌ | 仅文本 | ~51KB |
| Mix | 任务指令 + 完整HTML表格 | ✅ PNG | 文本+视觉 | ~51KB + 图片 |

**核心区别**：
- **Image-only**：测试纯视觉理解能力
- **Text-only**：测试纯文本解析能力（可选HTML/CSV/Markdown/LaTeX）
- **Mix**：测试多模态融合能力，看模型能否利用两种信息源互补

**数据字段使用**：
- ✅ 使用：`FileName`, `Question`, `QuestionType`, `SubQType`, `FinalAnswer`
- ❌ 未使用：`id`（仅用于跟踪）, `CompStrucCata`（元数据）, `Source`（元数据）, `COT`（人工标注，不发给API）
