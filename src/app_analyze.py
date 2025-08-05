import asyncio
import json
import re
from typing import List, Dict
from fastmcp import Client
from openai import OpenAI


class StarRocksAnalyzer:
    """StarRocks SQL分析与诊断助手"""

    def __init__(self, script: str, model: str = "gpt-4o-mini", max_tool_calls: int = 3):
        self.model = model
        self.mcp_client = Client(script)
        self.openai_client = OpenAI(
            base_url="https://api.agicto.cn/v1",
            api_key="sk-7o3jYn2HuoxJN7XcKA6FWxXVuZ3ZbIjfKLK7BPNLOfPP5ngO"
        )
        self.max_tool_calls = max_tool_calls  # 限制最大工具调用次数，防止无限递归
        self.current_tool_calls = 0  # 当前工具调用计数

        # 诊断prompt保持不变
        self.diagnosis_prompt = """
        # StarRocks SQL执行计划风险诊断专家
        ## 角色与任务
        作为资深StarRocks DBA，我将基于MCP执行计划的AI强化诊断引擎，严格分析SQL执行计划并诊断资源风险。

        ## 诊断推理流程
        ### 阶段1：物理计划指标提取
        1. **最耗时的操作节点** (Top Time-consuming Nodes)  
        2. **扫描节点关键指标** (OLAP_SCAN_NODE):  
           - 实际扫描效率  
           - 输出数据量  
        3. **网络传输瓶颈** (EXCHANGE):  
           - NetworkTime  
           - 数据倾斜迹象  
        4. **聚合/Join操作成本**:  
           - HASH_JOIN Build/Probe  
           - 实际输入输出rows 
           - PartitionExprs：收集分区键值
           - hash_join的内存使用量  
        5. **内存使用关键点**:  
           - 总内存消耗  
           - 实例峰值  
        6. **执行阶段分布** (时间占比):  
           - Scan  
           - Network  
           - ScheduleTime  

        ### 阶段2：逻辑计划指标提取
        ```python
        def extract_logical_metrics():
            olap_scan = {
                "predicates": "下推谓词列表", 
                "output_partition": "分区键值分布"
            }
            hash_join = {
                "join_type": "连接类型",
                "local_shuffle": "true/false"
            }
            return olap_scan, hash_join
        ```

        ### 阶段3：风险诊断（优先级排序）
        一、JOIN 顺序分析
        提取所有 HASH_JOIN 节点的 build/probe 输入行数
        判断是否存在多级 HASH_JOIN，且输入行数乘积 > 10^6
        若 local_shuffle = false，并存在大表连接，说明可能存在非本地 JOIN 传输瓶颈

        二、内存管控分析
        获取每个 OLAP_SCAN 节点的 predicates 数量与类型
        查看各节点的内存成本，重点检查 HASH_JOIN 节点的 memory_cost 或 peak_mem
        检查 PartitionExprs 字段粒度是否匹配后续 Hash Join 的建表字段

        三、其他优化方向诊断
        OLAP_SCAN 优化:
        若无谓词（predicates = null），且扫描效率 < 100MB/s，判断为无谓词低效扫描

        网络传输瓶颈:
        若 EXCHANGE 节点 NetworkTime 占总执行时间 > 40%，则为传输瓶颈
        若 EXCHANGE 节点各实例数据量标准差 > 均值 50%，则存在严重数据倾斜

        操作符高风险模式检测:
        无谓词全表扫描（SCAN 节点无 predicate）
        笛卡尔积连接（INNER JOIN 无连接条件）
        高基数聚合（AGGREGATE 预估输出行数 > 1 亿）

        ### 阶段4：优化建议生成
        if 内存超标：
            建议 += ["启用PARTITION PRUNING", "激活spill_to_disk"]
        if CPU超限 && 存在常量表达式：
            建议 += ["移除WHERE 1=1类无效条件"]
        if 扫描效率低下：
            建议 += ["添加Colocate Group", "创建Materialized View"]
        if 网络耗时高：
            建议 += ["改用BROADCAST JOIN", "调整shuffle粒度"]

        ### 输出要求
        - 按风险等级排序诊断结果（致命 > 高危 > 建议）
        - 每个结论必须关联物理 / 逻辑计划证据
        - 优化建议需标注预期收益指标
        - 必须使用中文输出
        """

        self.messages = [{
            "role": "system",
            "content": (
                "你是一个专业的 StarRocks 数据库分析助手。"
                "请严格遵循以下诊断流程分析 SQL 执行计划：\n"
                f"{self.diagnosis_prompt}"
            )
        }]
        self.tools = []
        self.tool_name_mapping = {}
        self.original_tools = {}

    async def prepare_tools(self):
        """准备并规范化工具列表"""
        try:
            tools = await self.mcp_client.list_tools()
            self.original_tools = {tool.name: tool for tool in tools}
            processed_tools = []
            for tool in tools:
                cleaned_name = re.sub(r'[^a-zA-Z0-9_-]', '-', tool.name)
                if not cleaned_name:
                    cleaned_name = f"tool-{id(tool)}"
                self.tool_name_mapping[cleaned_name] = tool.name
                processed_tools.append({
                    "type": "function",
                    "function": {
                        "name": cleaned_name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            return processed_tools
        except Exception as e:
            print(f"准备工具时出错: {str(e)}")
            return []

    def _process_tool_response(self, response):
        """统一处理工具响应"""
        if isinstance(response, list):
            texts = []
            for item in response:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif isinstance(item, str):
                    texts.append(item)
                else:
                    texts.append(str(item))
            return "\n".join(texts)
        elif hasattr(response, 'text'):
            return response.text
        return str(response)

    async def get_execution_plans(self, sql: str):
        """获取 SQL 的执行计划（物理和逻辑），当 explain_analyze 失败时提前退出"""
        physical_plan_response, logical_plan_response = None, None
        error_message = ""

        # 步骤 1: 单独获取物理计划 (EXPLAIN ANALYZE)
        try:
            physical_plan_response = await self.mcp_client.call_tool("explain_analyze", {"sql": sql})
            # 检查响应中是否包含表示错误的文本
            processed_physical_plan = self._process_tool_response(physical_plan_response)
            if "Getting analyzing error" in processed_physical_plan or "can not be analyzed" in processed_physical_plan:
                return {
                    "error": "UNPARSABLE_SQL",
                    "error_detail": processed_physical_plan,
                    "physical_plan": "",
                    "logical_plan": ""
                }
        except Exception as e:
            error_detail = self._process_tool_response(e)
            # 检查异常消息是否指示SQL无法解析
            if "Getting analyzing error" in error_detail or "can not be analyzed" in error_detail:
                return {
                    "error": "UNPARSABLE_SQL",
                    "error_detail": error_detail,
                    "physical_plan": "",
                    "logical_plan": ""
                }
            # 对于其他类型的异常，作为通用错误处理
            error_message = f"获取物理执行计划失败: {error_detail}"
            return {
                "error": error_message,
                "physical_plan": "",
                "logical_plan": ""
            }

        # 步骤 2: 如果物理计划成功，获取逻辑计划
        try:
            logical_plan_response = await self.mcp_client.call_tool("explain_verbose", {"sql": sql})
            print("逻辑与物理执行计划获取成功")
        except Exception as e:
            error_message = f"获取逻辑执行计划失败: {self._process_tool_response(e)}"
            # 即使逻辑计划失败，我们仍然可以尝试使用物理计划进行分析
            # 因此这里只记录错误，不立即返回

        return {
            "physical_plan": self._process_tool_response(physical_plan_response),
            "logical_plan": self._process_tool_response(logical_plan_response) if logical_plan_response else "",
            "error": error_message
        }

    async def analyze_sql(self, sql: str) -> str:
        """分析 SQL 语句，增强错误处理"""
        plans = await self.get_execution_plans(sql)
        
        if plans.get("error") == "UNPARSABLE_SQL":
            return f"无法解析SQL: {plans.get('error_detail', '')}\n分析已终止。"

        if plans["error"]:
            return f"SQL 分析无法进行：{plans['error']}\n请检查 SQL 语句的语法和完整性后重试。"
        
        analysis_request = {
            "role": "user",
            "content": (
                f"请分析以下 SQL 语句：\n{sql}\n\n"
                f"## 物理计划:\n{plans['physical_plan']}\n\n"
                f"## 逻辑计划:\n{plans['logical_plan']}\n\n"
                f"## 安全基线: {{\n\"MEMORY_LIMIT\": \"200MB\",\n\"CPU_LIMIT\": \"500ms\"\n}}\n\n"
                f"请严格按照诊断流程进行分析并输出结果。"
            )
        }
        self.messages.append(analysis_request)
        self.current_tool_calls = 0  # 重置
        response = await self.chat(self.messages)
        return self._process_tool_response(response)

    async def chat(self, messages: List[Dict]):
        """处理聊天交互，增加递归保护"""
        if not self.tools:
            self.tools = await self.prepare_tools()

        if self.current_tool_calls >= self.max_tool_calls:
            return {"role": "assistant", "content": f"已达到最大工具调用次数 ({self.max_tool_calls})，防止无限循环。"}

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            if not response.choices:
                return {"role": "assistant", "content": "未收到有效响应"}
            message = response.choices[0].message

            if not message.tool_calls:
                return message

            self.current_tool_calls += 1
            for tool_call in message.tool_calls:
                try:
                    original_name = self.tool_name_mapping.get(tool_call.function.name)
                    if not original_name:
                        raise ValueError(f"未知工具: {tool_call.function.name}")
                    args = json.loads(tool_call.function.arguments)
                    tool_response = await self.mcp_client.call_tool(original_name, args)
                    response_content = self._process_tool_response(tool_response)
                    self.messages.append({
                        "role": "function",
                        "name": tool_call.function.name,
                        "content": response_content
                    })
                except Exception as e:
                    error_msg = f"工具调用失败: {str(e)}"
                    self.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    return {"role": "assistant", "content": error_msg}

            if self.current_tool_calls < self.max_tool_calls:
                return await self.chat(self.messages)
            else:
                return {"role": "assistant", "content": f"已达到最大工具调用次数 ({self.max_tool_calls})，分析终止。"}

        except Exception as e:
            error_msg = f"处理出错: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return {"role": "assistant", "content": error_msg}

    async def interactive_loop(self):
        """交互式循环，完善退出机制"""
        try:
            async with self.mcp_client:
                print("StarRocks SQL 分析助手已就绪 (输入 exit 或 quit 退出)...")
                print("提示：请输入完整有效的 SQL 语句进行分析诊断")
                while True:
                    try:
                        question = input("\n 用户:").strip()
                        if question.lower() in ["exit", "quit"]:
                            print("再见！")
                            break
                        if not question:
                            print("请输入有效的 SQL 语句，或输入 exit 退出")
                            continue

                        response = await self.analyze_sql(question)

                        print("-" * 60)
                        print(response)
                        print("-" * 60)

                        # 如果是无法解析SQL的特定错误，则退出循环
                        if response.startswith("无法解析SQL:"):
                            print("\n由于SQL无法解析，分析会话结束。")
                            break

                    except KeyboardInterrupt:
                        print("\n 检测到中断请求，是否退出？(y/n)")
                        choice = input().strip().lower()
                        if choice in ["y", "yes"]:
                            print("再见！")
                            break
                        else:
                            print("继续分析，请输入 SQL 语句...")
                            continue
                    except Exception as e:
                        print(f"\n 发生错误: {str(e)}")
                        print("是否继续分析？(y/n)")
                        choice = input().strip().lower()
                        if choice not in ["y", "yes"]:
                            print("再见！")
                            break
        except Exception as e:
            print(f"程序启动失败: {str(e)}")
            return


if __name__ == "__main__":
    try:
        analyzer = StarRocksAnalyzer(
            script="/Users/wangyaosheng/Desktop/ifinD/code/mcp-server-starrocks/src/server_for_app.py",
            max_tool_calls=3
        )
        asyncio.run(analyzer.interactive_loop())
    except Exception as e:
        print(f"程序运行出错: {str(e)}")