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

网络中的节点不断收集新的交易打包到区块中，所有的交易会通过两两哈希的方式形成一个Merkle树，而打包的过程就是要完成工作量证明的要求，当节点解出了当前的随机数时，它就把当前的区块广播到其他所有节点，并且加到区块链上。

区块中的第一笔交易称之为CoinBase交易，是产生的新币，奖励给区块的产生者  
### CBlockHeader（部分）
```c++ 
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
### CBlock
继承自CBlockHeader，拥有其所有成员，作为所有交易的容器，
同时包含 fChecked 变量作为交易验证符号



### CBlockLocator
用于描述区块链中在其他节点的一个位置， 如果其他节点没有相同的分支，它可以找到一个最近的中继(最近的相同块)。 更进一步地讲，它可能是分叉前的一个位置


## Transaction
交易是比特币中的重要内容。源码在 bitcoin/src/private 中。

###  COutPut
功能为一个交易哈希值与输出下标的集合

###  CTxIn(部分)
 负责交易的输入，包括当前输入所对应上一笔交易的输出位置，
 并且还包括上一笔输出所需要的签名脚本

该模块为了实现所需的功能，定义了四个规则（已在代码中注释）

 1. 如果一笔交易中所有的SEQUENCE_FINAL都被赋值了相应的nSequence，那么nLockTime就会被禁用
 2. 如果设置了该值，nSequence不被用于相对时间锁定。规则1失效
 3. 如果规则1有效并且设置了此变量，那么相对锁定时间单位为512秒，否则锁定时间就为1个区块
 4. 如果nSequence用于相对时间锁，即规则1有效，那么这个变量就用来从nSequence计算对应的锁定时间

其中，相对时间锁粒度为了使用相同的位数来粗略地编码相同的挂钟时间。因为区块的产生限制于每600s产生一个，相对时间锁定的最小单位为512是，512 = 2^9，所以相对时间锁定的时间转化为相当于当前值左移9位

###  CTxOut
 负责交易的输出，该类中定义了输出金额和锁定脚本
###  CTransaction（部分）
该模块实现基本的交易，就是那些在网络中广播并被最终打包到区块中的数据结构。
 其中，一个交易可以包含多个交易输入和输出

 源代码中更改默认交易版本需要两个步骤：
  

 1. 首先通过碰撞MAX_STANDARD_VERSION来调整中继策略
 2. 然后在稍后的日期碰撞默认的CURRENT_VERSION   
    
最终MAX_STANDARD_VERSION和CURRENT_VERSION会一致

IsCoinBase 判断是否是创币交易
HasWitness判断交易是否有见证者
```c++ 
class CTransaction
{
public:
    
    static const int32_t CURRENT_VERSION=2;         //默认交易版本
    static const int32_t MAX_STANDARD_VERSION=2;    

    const std::vector<CTxIn> vin;       //交易输入
    const std::vector<CTxOut> vout;     //交易输出
    const int32_t nVersion;             //版本         
    const uint32_t nLockTime;           //锁定时间

private:
    const uint256 hash;
    uint256 ComputeHash() const;

public:
    CTransaction();
    /**可变交易转换为交易*/
    CTransaction(const CMutableTransaction &tx);
    CTransaction(CMutableTransaction &&tx);

    /*
    提供此反序列化构造函数而不是Unserialize方法。
    反序列化是不可能的，因为它需要覆盖const字段
    */
    template <typename Stream>
    CTransaction(deserialize_type, Stream& s) : CTransaction(CMutableTransaction(deserialize, s)) {}
    uint256 GetWitnessHash() const;         //计算包含交易和witness数据的散列           

    CAmount GetValueOut() const;            //返回交易出书金额总和      

    unsigned int GetTotalSize() const;      // 返回交易大小

};
```

###  CMutableTransaction
可变交易类，相比上文的CTransaction，主题内容大致相同，只是交易可以直接修改。在广播中传播和打包到区块的交易都是CTransaction类型。

##  TransactionPool
交易池作为鉴于区块和交易的数据结构，同样起着重要的作用。

当比特币网络把某个时刻产生的交易广播到网络时，矿工接收到交易后并不是立即打包到备选区块。而是将接收到的交易放到类似缓冲区的一个交易池里，然后会根据一定的优先顺序来选择交易打包，以此来保障自己能获得尽可能多的交易费。

该处代码位于 bitcoin/src/txmempool.h 

###  LockPoints
通过设定一个"假"的高度值，用来标识它们只存在于交易池中，从而实现交易锁定点的功能。锁定交易最后的区块高度和打包时间

###  CTxMemPoolEntry

 CTxMemPoolEntry 负责存储相应的交易，以及该交易对应的所有子孙交易
 
 当一个新的CTxMemPoolEntry被添加到交易池，会更新新添加交易的所有子孙交易的状态(包括子孙交易数量，大小，和交易费用）和祖父交易状态

并且，如果移除一个交易时，它对应所有的子孙交易也将同样被移除
```c++ 
class CTxMemPool;
//交易池基本构成元素
class CTxMemPoolEntry
{
private:
    CTransactionRef tx;         //交易引用
    CAmount nFee;               //交易费用            
    size_t nUsageSize;          //大小        
    int64_t nTime;              //交易时间戳   
    unsigned int entryHeight;   //区块高度 
    bool spendsCoinbase;        //上个交易是否是创币交易   
    int64_t feeDelta;           //交易优先级的一个标量  
    LockPoints lockPoints;      //锁定点，交易最后的区块高度和打包时间
    
    uint64_t nCountWithDescendants;     //子孙交易数量 
    uint64_t nSizeWithDescendants;      //大小      
    CAmount nModFeesWithDescendants;    //费用总和，包括当前交易  
    
    uint64_t nCountWithAncestors;       //祖先交易数量
    uint64_t nSizeWithAncestors;        //大小
    CAmount nModFeesWithAncestors;      //费用总和

public:
    CTxMemPoolEntry(const CTransactionRef& _tx, const CAmount& _nFee,
                    int64_t _nTime, unsigned int _entryHeight,
                    bool spendsCoinbase,
                    int64_t nSigOpsCost, LockPoints lp);
    // 更新子孙交易状态
    void UpdateDescendantState(int64_t modifySize, CAmount modifyFee, int64_t modifyCount);
    // 更新祖先交易状态
    void UpdateAncestorState(int64_t modifySize, CAmount modifyFee, int64_t modifyCount, int64_t modifySigOps);
    // 更新交易优先级
    void UpdateFeeDelta(int64_t feeDelta);
    // 更新锁定点
    void UpdateLockPoints(const LockPoints& lp);
};
```
其中GetCountWithDescendants，GetSizeWithDescendants，GetModFeesWithDescendants分别获取子孙交易信息

而GetCountWithAncestors，GetSizeWithAncestors，GetModFeesWithAncestors，GetSigOpCostWithAncestors分别获取祖先交易信息

 CTxMemPoolEntry还有不同的排序方法，应对不同的需求：

 1. CompareTxMemPoolEntryByDescendantScore，按score/size原则对CTxMemPoolEntry排序
 2. CompareTxMemPoolEntryByScore，按(fee+delta)/size原则对CTxMemPoolEntry排序
 3. CompareTxMemPoolEntryByEntryTime，按时间CTxMemPoolEntry对排序
 4. CompareTxMemPoolEntryByAncestorFee，按min(score/size of entry's tx, score/size with all ancestors)进行排序

### CTxMemPool（部分）
CTxMemPool 保存当前主链所有的交易。这些交易有可能被加入到下一个有效区块中

 当交易在比特币网络上广播时会被加入到交易池。
 比如以下新的交易将不会被加入到交易池中：
 *  没有满足最低交易费的交易
 *  "双花"交易
 *  一个非标准交易

该类存在sanity-check，check函数将保证pool的一致性。所有的输入都在mapNextTx数组里；sanity-check关闭，check函数无效。

更新交易时从上之下，先更新祖父节点信息。

 从mempool中移除一个交易集合：如果一个交易在这个集合中，那么它的所有子孙交易都必须在集合中，除非该交易已经被打包到区块中；如果要移除一个已经被打包到区块中的交易，那么要把updateDescendants设为true，从而更新mempool中所有子孙节点的祖先信息。

对于一个特定的交易，调用 removeUnchecked 之前，必须为同时为要移除的交易集合调用 UpdateForRemoveFromMempool 。使用每个 CTxMemPoolEntry 中 setMemPoolParents 来遍历要移除交易的祖先，这样能保证我们更新的正确性。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNzQxMTUwOTQsLTI5MjQyNjYwOSwxNT
k4NDc3MzE5LC0xMjg0MzM2ODI3LC0xNDQ1NTgyMTc0LC0xMjUy
MDQxNjkxLC05MTcxNzU1ODgsOTYyMTE1MjE4LC0xOTA0MzI2NT
MxLC0xOTY2NTY3MDY3LDcyNzY2MTk2NiwxNDE3NjM1MDk5LC03
MzUzODk1NzFdfQ==
-->