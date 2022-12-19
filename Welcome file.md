# Bitcoin源码分析

该文章对Bitcoin 0.21 版本进行粗略的开源代码分析，格式参见第五届开源代码创新大赛代码注解赛道。

## 主要程序

 1.   **Bitcoin Core.app** Bitcoin客户端图形界面
    
2.  **Bitcoind** /src/bitcoind Bitcoin命令行版,也是下一步源代码分析的重点(不能与Bitcoin Core同时运行）
    
3. **Bitcoin-cli** /src/bitcoin-cli Bitcoind的一个功能完备的RPC客户端，可以使用命令行查询某个区块信息，交易信息等等
    
4.  **Bitcoin-tx** /src/bitcoind Bitcoind的交易处理模块，可以进行交易的查询和创建

## 项目结构

首先根据Bitcoin的项目目录进行分析。
分成19个部分，每一个部分有各自的作用。

 1.  **bench**， 共识-法官判断
 2. **compact**， 兼容
 3. **config**， 配置
 4. **consensus**， 共识算法
 5. **crypto**， 加密解密
 6. **interfaces**， j事件，节点，钱包类
 7. **leveldb**，水平数据库
 8. **obj object**，cpp编译中间目录 
 9. **policy**， 策略
 10. **primitives**， 区块类，交易类
 11. **qt**， 图形界面类
 12. **rpc**， 通信
 13. **script**， 脚本
 14. **secp256k1**， 加密算法
 15. **support**，功能支持类
 16. **test**， 类的功能测试
 17. **univalue**，一致性 
 18. **wallet**， 钱包类
 19. **zmp**，通信信息 

<br>

接口可以分成三个部分：
 1. **WalletInitInterface.h**，钱包的抽象类接口 
 2. **validationinterface.h cpp**， 区块校验接口
 3. **ui_interface.h cpp**，图形界面的接口

<br>

运行模块分成四部分：
 1. **bitcoind**，bitcoin客户端核心模块 
 2. **bitcoin-cli**，RPC客户端 
 3. **bitcoin-tx**，交易处理模块
 4. **bitcoin-qt**，qt编写的图形化界面客户端 

<br>

运行模块分成四部分：


## BLOCK

区块是组成区块的基本单位，我们可以通过bitcoin-cli命令查看一个区块的基本信息。
在源代码中找一下区块的定义在primitives/block.h中:

### CBlockHeader
```c++ 
/*
网络中的节点不断收集新的交易打包到区块中，所有的交易会通过两两哈希的方式形成一个Merkle树
打包的过程就是要完成工作量证明的要求，当节点解出了当前的随机数时，
它就把当前的区块广播到其他所有节点，并且加到区块链上。
区块中的第一笔交易称之为CoinBase交易，是产生的新币，奖励给区块的产生者  
*/

class CBlockHeader
{
public:
    // header
    int32_t nVersion;       //版本
    uint256 hashPrevBlock;  //上一个区块的hash
    uint256 hashMerkleRoot; //包含交易信息的Merkle树根
    uint32_t nTime;         //时间戳
    uint32_t nBits;         //工作量证明(POW)的难度
    uint32_t nNonce;        //要找的符合POW的随机数

    CBlockHeader()          //构造函数初始化成员变量
    {
        SetNull();          
    }

    ADD_SERIALIZE_METHODS;  //通过封装的模板实现类的序列化

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(this->nVersion);
        READWRITE(hashPrevBlock);
        READWRITE(hashMerkleRoot);
        READWRITE(nTime);
        READWRITE(nBits);
        READWRITE(nNonce);
    }

    void SetNull()          //初始化成员变量
    {
        nVersion = 0;
        hashPrevBlock.SetNull();
        hashMerkleRoot.SetNull();
        nTime = 0;
        nBits = 0;
        nNonce = 0;
    }

    bool IsNull() const
    {
        return (nBits == 0);     //难度为0说明区块还未创建，区块头为空
    }

    uint256 GetHash() const;     //获取哈希

    int64_t GetBlockTime() const //获取区块时间
    {
        return (int64_t)nTime;
    }
};
```
### CBlock（部分）

```c++ 

class CBlock : public CBlockHeader         //继承自CBlockHeader，拥有其所有成员变量
{
public:
    // network and disk
    std::vector<CTransactionRef> vtx;      //所有交易的容器

    // memory only
    mutable bool fChecked;                 //交易是否验证
};
```

### CBlockLocator
用于描述区块链中在其他节点的一个位置， 如果其他节点没有相同的分支，它可以找到一个最近的中继(最近的相同块)。 更进一步地讲，它可能是分叉前的一个位置


## Transaction
交易是比特币中的重要内容。源码在 bitcoin/src/private 中。

### ### COutPut
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk3NTAzNjQxMCw5NjIxMTUyMTgsLTE5MD
QzMjY1MzEsLTE5NjY1NjcwNjcsNzI3NjYxOTY2LDE0MTc2MzUw
OTksLTczNTM4OTU3MV19
-->