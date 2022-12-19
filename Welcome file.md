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

###  COutPut
功能为一个交易哈希值与输出下标的集合

```c++ 
class COutPoint
{
public:
    uint256 hash;       //交易哈希
    uint32_t n;         //对应序列号

    COutPoint(): n((uint32_t) -1) { }       
    COutPoint(const uint256& hashIn, uint32_t nIn): hash(hashIn), n(nIn) { }

    ADD_SERIALIZE_METHODS;      //用来序列化数据结构，方便存储和传输

    template <typename Stream, typename Operation>
    inline void SerializationOp(Stream& s, Operation ser_action) {
        READWRITE(hash);
        READWRITE(n);
    }

    void SetNull() { hash.SetNull(); n = (uint32_t) -1; }
    bool IsNull() const { return (hash.IsNull() && n == (uint32_t) -1); }

    //<重载函数
    friend bool operator<(const COutPoint& a, const COutPoint& b)
    {
        int cmp = a.hash.Compare(b.hash);
        return cmp < 0 || (cmp == 0 && a.n < b.n);
    }

    //==重载函数
    friend bool operator==(const COutPoint& a, const COutPoint& b)
    {
        return (a.hash == b.hash && a.n == b.n);
    }

    //!=重载函数
    friend bool operator!=(const COutPoint& a, const COutPoint& b)
    {
        return !(a == b);
    }

    std::string ToString() const;
};
```
###  CTxIn(部分)
 负责交易的输入，包括当前输入所对应上一笔交易的输出位置，
 并且还包括上一笔输出所需要的签名脚本

该模块为了实现所需的功能，定义了四个规则（已在代码中注释）

```c++ 
class CTxIn
{
public:
    COutPoint prevout;      //上一笔交易输出位置
    CScript scriptSig;      //解锁脚本
    uint32_t nSequence;     //序列号，可用于交易的锁定 
                            
    CScriptWitness scriptWitness; 
    /* 
    规则1:如果一笔交易中所有的SEQUENCE_FINAL都被赋值了相应的nSequence，那么nLockTime就会被禁用
     */
    static const uint32_t SEQUENCE_FINAL = 0xffffffff;

    /* 
    规则2:如果设置了该值，nSequence不被用于相对时间锁定。规则1失效
     */
    static const uint32_t SEQUENCE_LOCKTIME_DISABLE_FLAG = (1 << 31);

    /* 
    规则3：如果规则1有效并且设置了此变量，那么相对锁定时间单位为512秒，否则锁定时间就为1个区块
     */
    static const uint32_t SEQUENCE_LOCKTIME_TYPE_FLAG = (1 << 22);

    /* 
    规则4：如果nSequence用于相对时间锁，即规则1有效，那么这个变量就用来从nSequence计算对应的锁定时间
     */
    static const uint32_t SEQUENCE_LOCKTIME_MASK = 0x0000ffff;

    /*
     相对时间锁粒度
     为了使用相同的位数来粗略地编码相同的挂钟时间，
     因为区块的产生限制于每600s产生一个，
     相对时间锁定的最小单位为512是，512 = 2^9
     所以相对时间锁定的时间转化为相当于当前值左移9位
     */
    static const int SEQUENCE_LOCKTIME_GRANULARITY = 9;

    CTxIn()
    {
        nSequence = SEQUENCE_FINAL;
    }

    explicit CTxIn(COutPoint prevoutIn, CScript scriptSigIn=CScript(), uint32_t nSequenceIn=SEQUENCE_FINAL);
    CTxIn(uint256 hashPrevTx, uint32_t nOut, CScript scriptSigIn=CScript(), uint32_t nSequenceIn=SEQUENCE_FINAL);

    ADD_SERIALIZE_METHODS;
};
```

###  CTxOut
 负责交易的输出，该类中定义了输出金额和锁定脚本
###  CTransaction（部分）
该模块实现基本的交易，就是那些在网络中广播并被最终打包到区块中的数据结构。
 其中，一个交易可以包含多个交易输入和输出
 
```c++ 
class CTransaction
{
public:
    
    static const int32_t CURRENT_VERSION=2;         //默认交易版本

    /* 
    更改默认交易版本需要两个步骤：
    1.首先通过碰撞MAX_STANDARD_VERSION来调整中继策略，
    2.然后在稍后的日期碰撞默认的CURRENT_VERSION   
    最终MAX_STANDARD_VERSION和CURRENT_VERSION会一致
    */
    static const int32_t MAX_STANDARD_VERSION=2;    

    /*
    下面这些变量都被定义为常量类型，从而避免无意识的修改了交易而没有更新缓存的hash值；
    然而CTransaction不是可变的
    反序列化和分配被执行的时候会绕过常量
    这才是安全的，因为更新整个结构包括哈希值
    */
    const std::vector<CTxIn> vin;       //交易输入
    const std::vector<CTxOut> vout;     //交易输出
    const int32_t nVersion;             //版本         
    const uint32_t nLockTime;           //锁定时间

private:
    /** Memory only. */
    const uint256 hash;

    uint256 ComputeHash() const;

public:
    /** Construct a CTransaction that qualifies as IsNull() */
    CTransaction();


    /**可变交易转换为交易*/
    CTransaction(const CMutableTransaction &tx);
    CTransaction(CMutableTransaction &&tx);

    template <typename Stream>
    inline void Serialize(Stream& s) const {
        SerializeTransaction(*this, s);
    }

    /*
    提供此反序列化构造函数而不是Unserialize方法。
    反序列化是不可能的，因为它需要覆盖const字段
    */
    template <typename Stream>
    CTransaction(deserialize_type, Stream& s) : CTransaction(CMutableTransaction(deserialize, s)) {}

    bool IsNull() const {
        return vin.empty() && vout.empty();
    }

    const uint256& GetHash() const {
        return hash;
    }
    uint256 GetWitnessHash() const;         //计算包含交易和witness数据的散列           

    // Return sum of txouts.
    CAmount GetValueOut() const;            //返回交易出书金额总和      
    // GetValueIn() is a method on CCoinsViewCache, because
    // inputs must be known to compute value in.

    unsigned int GetTotalSize() const;      // 返回交易大小

    bool IsCoinBase() const                 //判断是否是创币交易
    {
        return (vin.size() == 1 && vin[0].prevout.IsNull());
    }

    bool HasWitness() const
    {
        for (size_t i = 0; i < vin.size(); i++) {
            if (!vin[i].scriptWitness.IsNull()) {
                return true;
            }
        }
        return false;
    }
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
    size_t nTxWeight;           //          
    size_t nUsageSize;          //大小        
    int64_t nTime;              //交易时间戳   
    unsigned int entryHeight;   //区块高度 
    bool spendsCoinbase;        //上个交易是否是创币交易   
    int64_t sigOpCost;          //？？？  !< Total sigop cost
    int64_t feeDelta;           //交易优先级的一个标量  
    LockPoints lockPoints;      //锁定点，交易最后的区块高度和打包时间
    
    uint64_t nCountWithDescendants;     //子孙交易数量 
    uint64_t nSizeWithDescendants;      //大小      
    CAmount nModFeesWithDescendants;    //费用总和，包括当前交易  

    //祖先交易信息
    uint64_t nCountWithAncestors;       //祖先交易数量
    uint64_t nSizeWithAncestors;        //大小
    CAmount nModFeesWithAncestors;      //费用总和

public:
    CTxMemPoolEntry(const CTransactionRef& _tx, const CAmount& _nFee,
                    int64_t _nTime, unsigned int _entryHeight,
                    bool spendsCoinbase,
                    int64_t nSigOpsCost, LockPoints lp);

    const CTransaction& GetTx() const { return *this->tx; }
    CTransactionRef GetSharedTx() const { return this->tx; }
    const CAmount& GetFee() const { return nFee; }
    size_t GetTxSize() const;
    size_t GetTxWeight() const { return nTxWeight; }
    int64_t GetTime() const { return nTime; }
    unsigned int GetHeight() const { return entryHeight; }
    int64_t GetSigOpCost() const { return sigOpCost; }
    int64_t GetModifiedFee() const { return nFee + feeDelta; }
    size_t DynamicMemoryUsage() const { return nUsageSize; }
    const LockPoints& GetLockPoints() const { return lockPoints; }

    // Adjusts the descendant state.
    // 更新子孙交易状态
    void UpdateDescendantState(int64_t modifySize, CAmount modifyFee, int64_t modifyCount);
    // Adjusts the ancestor state
    // 更新祖先交易状态
    void UpdateAncestorState(int64_t modifySize, CAmount modifyFee, int64_t modifyCount, int64_t modifySigOps);
    // Updates the fee delta used for mining priority score, and the
    // modified fees with descendants.
    // 更新交易优先级
    void UpdateFeeDelta(int64_t feeDelta);
    // Update the LockPoints after a reorg
    // 更新锁定点
    void UpdateLockPoints(const LockPoints& lp);

    //获取子孙交易信息
    uint64_t GetCountWithDescendants() const { return nCountWithDescendants; }
    uint64_t GetSizeWithDescendants() const { return nSizeWithDescendants; }
    CAmount GetModFeesWithDescendants() const { return nModFeesWithDescendants; }

    bool GetSpendsCoinbase() const { return spendsCoinbase; }

    //获取祖先交易信息
    uint64_t GetCountWithAncestors() const { return nCountWithAncestors; }
    uint64_t GetSizeWithAncestors() const { return nSizeWithAncestors; }
    CAmount GetModFeesWithAncestors() const { return nModFeesWithAncestors; }
    int64_t GetSigOpCostWithAncestors() const { return nSigOpCostWithAncestors; }

    mutable size_t vTxHashesIdx;    //交易池哈希的下标  //!< Index in mempool's vTxHashes
};
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYzNjA5NjY4MCwtMTI1MjA0MTY5MSwtOT
E3MTc1NTg4LDk2MjExNTIxOCwtMTkwNDMyNjUzMSwtMTk2NjU2
NzA2Nyw3Mjc2NjE5NjYsMTQxNzYzNTA5OSwtNzM1Mzg5NTcxXX
0=
-->