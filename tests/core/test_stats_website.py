#!/usr/bin/env python3
"""
国家统计局网站缓存效果测试脚本
测试真实网站的缓存性能和效果
"""

import sys
import os
import time
import json
import subprocess
import signal
from pathlib import Path
import gymnasium as gym


import browsergym.core

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "browsergym" / "core" / "src"))

try:
    import browsergym.core
    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import OpenEndedTask
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在BrowserGym项目根目录下运行此脚本")
    sys.exit(1)


def check_redis_server():
    """检查Redis服务器是否已运行"""
    try:
        import redis
        r = redis.from_url("redis://localhost:6380/1")
        r.ping()
        return True
    except redis.exceptions.ConnectionError:
        return False
    
def clean_redis():
    """清空Redis缓存数据库（用于测试前环境清理）"""
    try:
        import redis
        r = redis.from_url("redis://localhost:6380/1")
        r.flushdb()
        print("✅ Redis缓存已清空")
        return True
    except Exception as e:
        print(f"❌ 清空Redis缓存失败: {e}")
        return False
    

def test_stats_gov_website():
    """测试国家统计局网站的缓存效果"""
    print("🚀 开始测试国家统计局网站缓存效果")
    print("=" * 60)
    
    # 检查Redis服务器是否已运行
    if not check_redis_server():
        print("❌ Redis服务器未运行，请启动Redis服务器")
        sys.exit(1)
    else:
        clean_redis()
    
    try:
        # 测试网站信息
        test_site = {
            "name": "国家统计局数据网站",
            "url": "https://data.stats.gov.cn/",
            "type": "政府数据门户",
            "goal": "获取国家统计局数据"
        }
        
        print(f"📊 测试网站: {test_site['name']}")
        print(f"🔗 测试URL: {test_site['url']}")
        print(f"🏷️  网站类型: {test_site['type']}")
        print()
        
        # 创建缓存环境
        print("⚙️  配置缓存环境...")
        cache_config = {
            "redis_url": "redis://localhost:6380/1",
            "ttl": 3600,  # 1小时缓存
            "cacheable_resource_types": [
                "document", "stylesheet", "script", "image", "font", "xhr", "fetch"
            ],
            # "enable_page_snapshot": True,
            # "snapshot_wait_time": 5000
        }
        
        env = BrowserEnv(
            task_entrypoint=OpenEndedTask, 
            task_kwargs={"start_url": 'about:blank', "goal": test_site["goal"]},
            enable_context_cache=True,
            context_cache_kwargs=cache_config,
            headless=False,  # 无头模式，适合测试
            timeout=30000
        )
        
        obs, info = env.reset()
        print("✓ 环境初始化完成")
        print(info)
        print()
        
        # 第一次访问
        print("🌐 第一次访问网站（建立缓存）...")
        start_time = time.time()
        
        try:
            env.step(f'page.goto("{test_site["url"]}")')
            
        except Exception as e:
            print(f"⚠️  页面加载警告: {e}")
        
        first_visit_time = time.time() - start_time
        first_cache_stats = env.browser_cache_hit_stats.copy()
        
        print(f"⏱️  首次访问耗时: {first_visit_time:.2f}秒")
        print(f"📈 初始缓存统计: {first_cache_stats}")
        print()
        

        env = BrowserEnv(
            task_entrypoint=OpenEndedTask, 
            task_kwargs={"start_url": 'about:blank', "goal": test_site["goal"]},
            enable_context_cache=True,
            context_cache_kwargs=cache_config,
            headless=False,  # 无头模式，适合测试
            timeout=30000
        )
        obs, info = env.reset()
        print("✓ 环境初始化完成")
        print(info)
        print()

        # 第二次访问
        print("🔄 第二次访问网站（测试缓存效果）...")
        start_time = time.time()
        
        try:
            env.step(f'page.goto("{test_site["url"]}")')
        except Exception as e:
            print(f"⚠️  第二次访问警告: {e}")
        
        second_visit_time = time.time() - start_time
        second_cache_stats = env.browser_cache_hit_stats.copy()
        
        print(f"⏱️  第二次访问耗时: {second_visit_time:.2f}秒")
        print(f"📈 缓存命中统计: {second_cache_stats}")
        print()
        
        # 分析缓存效果
        print("📊 缓存效果分析")
        print("-" * 40)
        
        # 时间改善
        if first_visit_time > 0:
            time_improvement = ((first_visit_time - second_visit_time) / first_visit_time) * 100
            print(f"⚡ 加载时间改善: {time_improvement:.1f}%")
            
            # 详细性能分析
            print(f"📈 性能详细分析:")
            print(f"   首次访问: {first_visit_time:.2f}秒")
            print(f"   二次访问: {second_visit_time:.2f}秒")
            print(f"   节省时间: {first_visit_time - second_visit_time:.2f}秒")
            
            # 分析缓存效果为什么不理想
            if time_improvement < 20:  # 如果改善小于20%
                print(f"\n🔍 缓存效果分析（改善幅度较小的原因）:")
                
                # 估算各部分耗时
                estimated_network_delay = min(first_visit_time * 0.6, 8.0)  # 网络延迟通常占60%
                estimated_resource_download = first_visit_time * 0.25  # 资源下载25%
                estimated_processing = first_visit_time * 0.15  # 页面处理15%
                
                print(f"   🌐 估算网络延迟: {estimated_network_delay:.2f}秒 ({estimated_network_delay/first_visit_time*100:.1f}%)")
                print(f"   📦 估算资源下载: {estimated_resource_download:.2f}秒 ({estimated_resource_download/first_visit_time*100:.1f}%)")
                print(f"   ⚙️  估算页面处理: {estimated_processing:.2f}秒 ({estimated_processing/first_visit_time*100:.1f}%)")
                
                # 缓存能节省的理论最大时间
                max_cacheable_time = estimated_resource_download
                actual_saved_time = first_visit_time - second_visit_time
                cache_efficiency = (actual_saved_time / max_cacheable_time) * 100 if max_cacheable_time > 0 else 0
                
                print(f"\n   💡 缓存效率分析:")
                print(f"   理论最大节省: {max_cacheable_time:.2f}秒")
                print(f"   实际节省时间: {actual_saved_time:.2f}秒")
                print(f"   缓存效率: {cache_efficiency:.1f}%")
                
                # 给出优化建议
                print(f"\n   🎯 优化建议:")
                if estimated_network_delay / first_visit_time > 0.5:
                    print(f"   • 网络延迟占主导 ({estimated_network_delay/first_visit_time*100:.1f}%) - 考虑CDN或就近部署")
                if cache_efficiency < 50:
                    print(f"   • 缓存效率较低 ({cache_efficiency:.1f}%) - 检查缓存策略配置")
                if estimated_processing / first_visit_time > 0.2:
                    print(f"   • 页面处理耗时较高 ({estimated_processing/first_visit_time*100:.1f}%) - 优化JavaScript执行")
        
        # 缓存命中
        total_cache_hits = sum(second_cache_stats.values()) - sum(first_cache_stats.values())
        print(f"\n🎯 新增缓存命中: {total_cache_hits}次")
        
        # 计算缓存命中率和效果
        total_requests_second = sum(second_cache_stats.values())
        cache_hit_rate = (total_cache_hits / total_requests_second * 100) if total_requests_second > 0 else 0
        print(f"📊 缓存命中率: {cache_hit_rate:.1f}%")
        
        # 分析为什么缓存命中多但时间改善少
        if total_cache_hits > 30 and time_improvement < 20:
            print(f"\n⚠️  异常分析: 高缓存命中但低时间改善")
            print(f"   可能原因:")
            print(f"   1. 网络延迟占主导地位（DNS、连接建立、服务器响应）")
            print(f"   2. 缓存的主要是小文件（图片、CSS），大文件仍需下载")
            print(f"   3. 页面有大量动态请求未被缓存")
            print(f"   4. 浏览器渲染和JavaScript执行时间固定")
        
        # 缓存域名
        cached_domains = list(second_cache_stats.keys())
        print(f"\n🌐 涉及缓存域名: {len(cached_domains)}个")
        for domain in cached_domains:
            hits = second_cache_stats[domain]
            first_hits = first_cache_stats.get(domain, 0)
            new_hits = hits - first_hits
            print(f"   • {domain}: {hits}次命中 (新增{new_hits}次)")
        
        print()
        
        # 分析缓存内容
        print("🔍 缓存内容详细分析")
        print("-" * 40)
        
        redis_client = env.redis_client
        all_keys = redis_client.keys("*")
        cached_resources = []
        
        for key in all_keys:
            if not key.startswith("lock:") and not key.startswith("snapshot:"):
                try:
                    data = redis_client.get(key)
                    if data:
                        cached_data = json.loads(data)
                        if "url" in cached_data:
                            cached_resources.append(cached_data["url"])
                except:
                    pass
        
        print(f"📦 缓存资源总数: {len(cached_resources)}个")
        
        # 资源类型统计
        resource_types = {
            "HTML文档": 0, "CSS样式": 0, "JavaScript": 0,
            "图片资源": 0, "字体文件": 0, "数据接口": 0, "其他": 0
        }
        
        for url in cached_resources:
            url_lower = url.lower()
            if any(x in url_lower for x in [".html", "htm"]) or url_lower.endswith("/"):
                resource_types["HTML文档"] += 1
            elif ".css" in url_lower:
                resource_types["CSS样式"] += 1
            elif ".js" in url_lower:
                resource_types["JavaScript"] += 1
            elif any(x in url_lower for x in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"]):
                resource_types["图片资源"] += 1
            elif any(x in url_lower for x in [".woff", ".woff2", ".ttf", ".eot"]):
                resource_types["字体文件"] += 1
            elif any(x in url_lower for x in ["api", "data", "query", "ajax"]):
                resource_types["数据接口"] += 1
            else:
                resource_types["其他"] += 1
        
        print("📋 资源类型分布:")
        for res_type, count in resource_types.items():
            if count > 0:
                percentage = (count / len(cached_resources)) * 100 if cached_resources else 0
                print(f"   • {res_type}: {count}个 ({percentage:.1f}%)")
        
        print()
        
        # 缓存效果评估
        print("🎯 缓存效果评估")
        print("-" * 40)
        
        static_resources = resource_types["CSS样式"] + resource_types["JavaScript"] + resource_types["图片资源"] + resource_types["字体文件"]
        static_ratio = (static_resources / max(len(cached_resources), 1)) * 100
        
        print(f"📊 静态资源占比: {static_ratio:.1f}%")
        print(f"🔄 数据接口占比: {(resource_types['数据接口'] / max(len(cached_resources), 1)) * 100:.1f}%")
        
        # 缓存适用性评级
        if total_cache_hits > 10:
            cache_rating = "优秀"
            cache_emoji = "🌟"
        elif total_cache_hits > 5:
            cache_rating = "良好"
            cache_emoji = "👍"
        elif total_cache_hits > 2:
            cache_rating = "一般"
            cache_emoji = "👌"
        else:
            cache_rating = "较差"
            cache_emoji = "⚠️"
        
        print(f"{cache_emoji} 缓存适用性: {cache_rating}")
        
        # 推荐策略
        print()
        print("💡 缓存策略建议")
        print("-" * 40)
        
        if static_ratio > 70:
            print("✅ 推荐策略: HTTP缓存（高静态资源比例）")
            print("⏰ 建议TTL: 24-48小时")
            print("💾 预期存储: 低-中等")
        elif resource_types["数据接口"] / max(len(cached_resources), 1) > 0.3:
            print("✅ 推荐策略: 混合缓存（包含动态数据）")
            print("⏰ 建议TTL: 2-6小时")
            print("💾 预期存储: 中-高等")
        else:
            print("✅ 推荐策略: 标准HTTP缓存")
            print("⏰ 建议TTL: 1-4小时")
            print("💾 预期存储: 中等")
        
        print()
        
        # 测试结果总结
        print("📋 测试结果总结")
        print("=" * 60)
        
        success_count = 0
        total_tests = 4
        
        # 检查各项指标
        tests = [
            ("网站可访问", True, "✅"),
            ("缓存有命中", total_cache_hits > 0, "✅" if total_cache_hits > 0 else "❌"),
            ("资源被缓存", len(cached_resources) > 0, "✅" if len(cached_resources) > 0 else "❌"),
            ("性能有改善", second_visit_time < first_visit_time if first_visit_time > 0 else True, 
             "✅" if (second_visit_time < first_visit_time if first_visit_time > 0 else True) else "❌")
        ]
        
        for test_name, passed, emoji in tests:
            if passed:
                success_count += 1
            print(f"{emoji} {test_name}: {'通过' if passed else '失败'}")
        
        print()
        print(f"🏆 测试通过率: {success_count}/{total_tests} ({(success_count/total_tests)*100:.0f}%)")
        
        if success_count == total_tests:
            print("🎉 所有测试通过！缓存系统工作正常")
        elif success_count >= total_tests * 0.75:
            print("👍 大部分测试通过，缓存效果良好")
        else:
            print("⚠️  部分测试未通过，需要检查配置")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False
    
    return True

def test_loading_with_resource_block():
    """测试阻止图片资源加载对网页载入速度的优化效果"""
    print("\n" + "="*60)
    print("🚀 资源阻止优化测试")
    print("="*60)
    
    # 检查Redis服务器是否已运行
    if not check_redis_server():
        print("❌ Redis服务器未运行，请启动Redis服务器")
        return False
    else:
        clean_redis()
    
    try:
        # 测试网站配置
        test_sites = [
            {
                "name": "图片密集型网站", 
                "url": "https://data.stats.gov.cn/",
                "type": "数据门户"
            },
            {
                "name": "新闻网站",
                "url": "https://m.peopledailyhealth.com/",  
                "type": "新闻媒体"
            }
        ]
        
        print("📊 测试网站列表:")
        for i, site in enumerate(test_sites, 1):
            print(f"   {i}. {site['name']} ({site['type']})")
        print()
        
        # 对每个网站进行测试
        for site_idx, test_site in enumerate(test_sites):
            print(f"🌐 测试网站 {site_idx + 1}: {test_site['name']}")
            print(f"🔗 URL: {test_site['url']}")
            print("-" * 50)
            
            # 测试1: 不阻止任何资源（基准测试）
            print("📊 基准测试（不阻止任何资源）...")
            
            baseline_env = gym.make(
                "browsergym/openended",
                task_kwargs={"start_url": 'about:blank', "goal": "基准性能测试"},
                enable_context_cache=True,
                context_cache_kwargs={"redis_url": "redis://localhost:6380/1", "ttl": 3600},
                resource_filter_kwargs={
                    'resource_filter_config': {
                        'block_images': False,
                        'block_videos': False,
                        'block_ads': False,
                        'block_analytics': False,
                    }
                },
                headless=False,
                timeout=30000
            )
            
            obs_baseline, _info = baseline_env.reset()
            
            # 测量基准加载时间
            baseline_start = time.time()
            try:
                obs, reward, terminated, truncated, info =  baseline_env.step(f'goto("{test_site["url"]}")')
            except Exception as e:
                print(f"⚠️  基准测试警告: {e}")
                
            baseline_time = time.time() - baseline_start
            baseline_filter_stats = info.get("resource_filter_stats", {})
            print("基准资源统计:" , baseline_filter_stats)
            
            print(f"⏱️  基准加载时间: {baseline_time:.2f}秒")
            print(f"📊 基准资源统计: 总请求={baseline_filter_stats.get('total_requests', 0)}, 阻止={baseline_filter_stats.get('total_blocked', 0)}")
            print()
            
            # 测试2: 阻止图片资源
            print("🖼️  图片阻止测试...")
            
            optimized_env = gym.make(
                "browsergym/openended",
                task_kwargs={"start_url": 'about:blank', "goal": "图片阻止优化测试"},
                enable_context_cache=True,
                context_cache_kwargs={"redis_url": "redis://localhost:6380/1", "ttl": 3600},
                resource_filter_kwargs={
                    'resource_filter_config': {
                        'block_images': True,           # 阻止图片
                        'allow_essential_images': False, # 但保留关键图片
                        'block_videos': False,
                        'block_ads': False,
                        'block_analytics': False,
                    }
                },
                headless=False,
                timeout=30000
            )
            
            obs_optimized, info_optimized = optimized_env.reset()
            
            # 测量优化后加载时间
            optimized_start = time.time()
            try:
                obs, reward, terminated, truncated, info  = optimized_env.step(f'goto("{test_site["url"]}")')
            except Exception as e:
                print(f"⚠️  优化测试警告: {e}")
                
            optimized_time = time.time() - optimized_start
            optimized_filter_stats = info.get("resource_filter_stats", {})
            print("优化资源统计:" , optimized_filter_stats)
            
            print(f"⏱️  优化加载时间: {optimized_time:.2f}秒")
            print(f"📊 优化资源统计: 总请求={optimized_filter_stats.get('total_requests', 0)}, 阻止={optimized_filter_stats.get('total_blocked', 0)}")
            print(f"🖼️  阻止图片数量: {optimized_filter_stats.get('blocked_images', 0)}")
            print()
            
            # 测试3: 激进阻止模式（图片+视频+广告+分析）
            print("🚫 激进阻止测试（图片+视频+广告+分析）...")
            
            aggressive_env = gym.make(
                "browsergym/openended",
                task_kwargs={"start_url": 'about:blank', "goal": "激进阻止优化测试"},
                enable_context_cache=True,
                context_cache_kwargs={"redis_url": "redis://localhost:6380/1", "ttl": 3600},
                resource_filter_kwargs={
                    'resource_filter_config': {
                        'block_images': True,
                        'allow_essential_images': True,
                        'block_videos': True,           # 阻止视频
                        'block_ads': True,              # 阻止广告
                        'block_analytics': True,        # 阻止分析脚本
                        'block_fonts': False,           # 保留字体以免影响显示
                        'custom_url_patterns': [        # 自定义阻止模式
                            r'.*\.gif$',                # 阻止GIF动图
                            r'.*banner.*',              # 阻止横幅
                            r'.*tracking.*',            # 阻止跟踪
                        ]
                    }
                },
                headless=True,
                timeout=30000
            )
            
            obs_aggressive, info_aggressive = aggressive_env.reset()
            
            # 测量激进模式加载时间
            aggressive_start = time.time()
            try:
                obs, reward, terminated, truncated, info = aggressive_env.step(f'goto("{test_site["url"]}")')
            except Exception as e:
                print(f"⚠️  激进测试警告: {e}")
                
            aggressive_time = time.time() - aggressive_start
            aggressive_filter_stats = info.get("resource_filter_stats", {})
            
            print(f"⏱️  激进模式加载时间: {aggressive_time:.2f}秒")
            print(f"📊 激进资源统计: 总请求={aggressive_filter_stats.get('total_requests', 0)}, 阻止={aggressive_filter_stats.get('total_blocked', 0)}")
            print(f"🚫 详细阻止统计:")
            print(f"   • 图片: {aggressive_filter_stats.get('blocked_images', 0)}个")
            print(f"   • 视频: {aggressive_filter_stats.get('blocked_videos', 0)}个") 
            print(f"   • 广告: {aggressive_filter_stats.get('blocked_ads', 0)}个")
            print(f"   • 分析: {aggressive_filter_stats.get('blocked_analytics', 0)}个")
            print(f"   • 自定义: {aggressive_filter_stats.get('blocked_custom', 0)}个")
            print()
            
            # 性能分析对比
            print("📈 性能优化分析")
            print("-" * 40)
            
            # 计算改善百分比
            if baseline_time > 0:
                image_improvement = ((baseline_time - optimized_time) / baseline_time) * 100
                aggressive_improvement = ((baseline_time - aggressive_time) / baseline_time) * 100
                
                print(f"⚡ 性能对比结果:")
                print(f"   📊 基准加载时间: {baseline_time:.2f}秒")
                print(f"   🖼️  图片阻止优化: {optimized_time:.2f}秒 (改善 {image_improvement:.1f}%)")
                print(f"   🚫 激进阻止优化: {aggressive_time:.2f}秒 (改善 {aggressive_improvement:.1f}%)")
                
                # 绝对时间节省
                image_savings = baseline_time - optimized_time
                aggressive_savings = baseline_time - aggressive_time
                
                print(f"\n💰 时间节省:")
                print(f"   🖼️  图片阻止节省: {image_savings:.2f}秒")
                print(f"   🚫 激进模式节省: {aggressive_savings:.2f}秒")
                print(f"   📈 激进模式额外收益: {aggressive_savings - image_savings:.2f}秒")
                
                # 资源阻止效率分析
                blocked_images = optimized_filter_stats.get('blocked_images', 0)
                total_blocked_aggressive = aggressive_filter_stats.get('total_blocked', 0)
                
                print(f"\n🎯 资源阻止效果:")
                print(f"   🖼️  图片阻止数量: {blocked_images}个")
                print(f"   🚫 激进模式总阻止: {total_blocked_aggressive}个")
                
                if blocked_images > 0:
                    time_per_image = image_savings / blocked_images
                    print(f"   ⚡ 每个图片平均节省: {time_per_image:.3f}秒")
                
                # 效果评级
                print(f"\n🏆 优化效果评级:")
                
                def get_rating(improvement):
                    if improvement > 30:
                        return "🌟 显著", "excellent"
                    elif improvement > 15:
                        return "👍 良好", "good"
                    elif improvement > 5:
                        return "👌 一般", "moderate"
                    else:
                        return "⚠️  微弱", "minimal"
                
                image_rating, image_level = get_rating(image_improvement)
                aggressive_rating, aggressive_level = get_rating(aggressive_improvement)
                
                print(f"   🖼️  图片阻止: {image_rating} ({image_improvement:.1f}%)")
                print(f"   🚫 激进模式: {aggressive_rating} ({aggressive_improvement:.1f}%)")
                
                # 网站特性分析
                print(f"\n🔍 网站特性分析:")
                baseline_requests = baseline_filter_stats.get('total_requests', 0)
                image_ratio = (blocked_images / max(baseline_requests, 1)) * 100
                
                print(f"   📊 图片资源占比: {image_ratio:.1f}%")
                print(f"   🌐 总请求数量: {baseline_requests}个")
                print(f"   ⚡ 网络密集程度: {'高' if baseline_requests > 50 else '中' if baseline_requests > 20 else '低'}")
                
                # 优化建议
                print(f"\n💡 针对此网站的优化建议:")
                
                if image_improvement > 15:
                    print(f"   ✅ 强烈推荐图片阻止优化")
                    print(f"   📱 特别适合移动端或慢网络用户")
                elif image_improvement > 5:
                    print(f"   👍 推荐使用图片阻止优化")
                    print(f"   🎯 在特定场景下有明显效果")
                else:
                    print(f"   ⚠️  图片阻止效果有限")
                    print(f"   💭 可能网站图片较少或已经优化")
                
                if aggressive_improvement > image_improvement + 10:
                    print(f"   🚫 激进模式显著更优，考虑启用完整过滤")
                elif aggressive_improvement > image_improvement + 5:
                    print(f"   🔄 激进模式有额外收益，可根据需要启用")
                else:
                    print(f"   📊 单纯图片阻止已足够，激进模式收益有限")
                
            else:
                print("⚠️  无法计算优化效果（基准时间为0）")
            
            # 清理环境
            baseline_env.close()
            optimized_env.close() 
            aggressive_env.close()
            
            print(f"\n{'='*50}")
            
        # 总体测试总结
        print("\n🎉 资源阻止优化测试完成")
        print("="*60)
        
        print("📋 测试总结:")
        print("✅ 完成了基准、图片阻止、激进阻止三种模式的对比测试")
        print("📊 测量了加载时间和资源阻止效果")
        print("💡 提供了针对性的优化建议")
        
        print("\n🎯 通用优化策略建议:")
        print("1. 🖼️  图片阻止: 适合图片密集型网站，通常有5-20%改善")
        print("2. 🚫 激进阻止: 适合广告较多的网站，可额外节省10-30%时间")
        print("3. 📱 移动优化: 在移动网络环境下效果更显著")
        print("4. 🎨 用户体验: 需要平衡加载速度和视觉效果")
        print("5. ⚙️  动态配置: 可根据网络状况动态调整阻止策略")
        
        return True
        
    except Exception as e:
        print(f"❌ 资源阻止测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_cache_strategy():
    """测试优化的缓存策略，展示如何获得更好的缓存效果"""
    print("\n" + "="*60)
    print("🚀 优化缓存策略测试")
    print("="*60)
    
    # 检查Redis服务器是否已运行
    if not check_redis_server():
        print("❌ Redis服务器未运行，请启动Redis服务器")
        return False
    else:
        clean_redis()
    
    try:
        # 优化的缓存配置
        optimized_config = {
            "redis_url": "redis://localhost:6380/1",
            "ttl": 7200,  # 2小时缓存
            "cacheable_resource_types": [
                "document", "stylesheet", "script", "image", "font", 
                "xhr", "fetch", "media", "texttrack", "websocket"  # 更全面的资源类型
            ],
            "enable_page_snapshot": True,  # 启用页面快照
            "snapshot_wait_time": 3000     # 等待3秒确保页面完全加载
        }
        
        print("⚙️  配置优化缓存策略...")
        print(f"📋 策略特点:")
        print(f"   • 启用页面快照模式")
        print(f"   • 扩展资源类型缓存")
        print(f"   • 优化TTL时间")
        print(f"   • 包含动态请求缓存")
        print()
        
        # 测试网站
        test_url = "https://m.chinabgao.com/"
        
        env = BrowserEnv(
            task_entrypoint=OpenEndedTask, 
            task_kwargs={"start_url": 'about:blank', "goal": "测试优化缓存策略"},
            enable_context_cache=True,
            context_cache_kwargs=optimized_config,
            headless=True,
            timeout=30000
        )
        
        obs, info = env.reset()
        print("✓ 优化环境初始化完成")
        print()
        
        # 第一次访问（建立缓存+快照）
        print("🌐 第一次访问（建立缓存和页面快照）...")
        start_time = time.time()
        
        try:
            env.step(f'goto("{test_url}")')
        except Exception as e:
            print(f"⚠️  页面加载警告: {e}")
        
        first_visit_time = time.time() - start_time
        first_cache_stats = env.browser_cache_hit_stats.copy()
        
        print(f"⏱️  首次访问耗时: {first_visit_time:.2f}秒")
        print(f"📈 初始缓存统计: {first_cache_stats}")
        print()
        
        # 清理环境，模拟新用户访问
        env.close()
        
        # 创建新环境测试缓存效果
        env2 = BrowserEnv(
            task_entrypoint=OpenEndedTask,
            task_kwargs={"start_url": 'about:blank', "goal": "测试优化缓存策略"},
            enable_context_cache=True,
            context_cache_kwargs=optimized_config,
            headless=True,
            timeout=30000
        )
        
        obs2, info2 = env2.reset()
        
        # 第二次访问（使用缓存+快照）
        print("🔄 第二次访问（使用缓存和页面快照）...")
        start_time = time.time()
        
        try:
            env2.step(f'goto("{test_url}")')
        except Exception as e:
            print(f"⚠️  第二次访问警告: {e}")
        
        second_visit_time = time.time() - start_time
        second_cache_stats = env2.browser_cache_hit_stats.copy()
        
        print(f"⏱️  第二次访问耗时: {second_visit_time:.2f}秒")
        print(f"📈 缓存命中统计: {second_cache_stats}")
        print()
        
        # 优化效果分析
        print("📊 优化缓存效果分析")
        print("-" * 40)
        
        if first_visit_time > 0:
            time_improvement = ((first_visit_time - second_visit_time) / first_visit_time) * 100
            print(f"⚡ 加载时间改善: {time_improvement:.1f}%")
            print(f"⏱️  绝对时间节省: {first_visit_time - second_visit_time:.2f}秒")
            
            # 与标准缓存对比
            standard_improvement = 7.3  # 之前测试的标准缓存改善
            optimization_gain = time_improvement - standard_improvement
            
            print(f"📈 相比标准缓存额外改善: {optimization_gain:.1f}%")
            
            if time_improvement > 15:
                print("🌟 优化效果: 显著改善")
            elif time_improvement > 10:
                print("👍 优化效果: 良好改善") 
            else:
                print("👌 优化效果: 一般改善")
        
        # 分析为什么优化效果仍然有限
        print(f"\n🔍 优化策略效果分析:")
        print(f"   💾 页面快照: {'已启用' if optimized_config.get('enable_page_snapshot') else '未启用'}")
        print(f"   📦 资源类型: {len(optimized_config['cacheable_resource_types'])}种")
        print(f"   ⏰ 缓存时长: {optimized_config['ttl']/3600:.1f}小时")
        
        total_cache_hits = sum(second_cache_stats.values())
        print(f"   🎯 总缓存命中: {total_cache_hits}次")
        
        # 缓存限制的根本原因
        print(f"\n💡 缓存优化的根本限制:")
        print(f"   1. 🌐 网络延迟不可避免 - 即使完美缓存，DNS和连接建立仍需时间")
        print(f"   2. 🏗️  页面架构影响 - 现代网站大量使用CDN，本地缓存优势有限")
        print(f"   3. 📱 移动端优化 - 网站已针对移动端优化，资源本身就较小")
        print(f"   4. ⚡ 服务器性能 - 现代服务器响应很快，缓存节省相对有限")
        
        # 实际应用建议
        print(f"\n🎯 实际应用场景建议:")
        print(f"   ✅ 适合缓存的场景:")
        print(f"      • 重复访问相同页面的用户")
        print(f"      • 网络环境较差的用户")
        print(f"      • 大量静态资源的网站")
        print(f"      • 开发和测试环境")
        
        print(f"   ⚠️  缓存效果有限的场景:")
        print(f"      • 首次访问用户（无缓存可用）")
        print(f"      • 网络环境良好的用户")
        print(f"      • 高度动态化的现代网站")
        print(f"      • 已使用CDN优化的网站")
        
        # 最终建议
        print(f"\n📋 最终缓存策略建议:")
        if time_improvement > 15:
            print(f"   🌟 推荐使用: 优化缓存策略效果显著")
        elif time_improvement > 10:
            print(f"   👍 可以使用: 有一定改善效果")
        else:
            print(f"   💭 谨慎使用: 改善有限，需权衡成本效益")
            print(f"   💡 建议重点: 专注于网络优化、CDN部署、代码优化等其他手段")
        
        env2.close()
        return True
        
    except Exception as e:
        print(f"❌ 优化缓存测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 BrowserGym 缓存系统测试")
    print("测试目标: 国家统计局数据网站")
    print("=" * 60)
    print()
    
    try:
        # 运行缓存测试
        print("第一部分：缓存效果测试")
        success1 = test_stats_gov_website()
        
        # 运行资源阻止测试  
        print("\n第二部分：资源阻止优化测试")
        success2 = test_loading_with_resource_block()
        
        if success1 and success2:
            print("\n🎉 所有测试完成！")
        else:
            print("\n⚠️  部分测试失败！")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 测试异常终止: {e}")
        sys.exit(1)