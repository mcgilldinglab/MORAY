// Fast multi-threaded MPHF read-query. Bit-identical to score_k25_level_masks (species level),
// but: reads loaded in memory, OpenMP over reads, thread-local FLAT arrays (no per-read hashmap).
// Compile: g++ -O3 -std=c++17 -pthread -fopenmp mphf_query_fast.cpp -o mphf_query_fast
#include "BooPHF.h"
#include "mphf_format.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;
using u64=uint64_t; using u32=uint32_t; using u16=uint16_t;
typedef boomphf::SingleHashFunctor<u64> hasher_t;
typedef boomphf::mphf<u64, hasher_t> boophf_t;
static u64 gv(const uint8_t* p,size_t& i,size_t lim){ u64 v=0;int s=0;uint8_t c; do{ if(i>=lim)return v; c=p[i++]; v|=(u64)(c&0x7f)<<s; s+=7;}while(c&0x80); return v; }

struct Index {
    boophf_t bphf; u64 n=0; std::vector<u32> fp, payload; std::vector<uint8_t> blob; bool loaded=false;
    void load(const std::string& d){
        std::ifstream is(d+"/meta.bin",std::ios::binary);
        MphfHeader h{};
        if(!is.read((char*)&h,sizeof(h))) throw std::runtime_error("meta.bin too short: "+d);
        if(const char* why = mphf_check_header(h)) throw std::runtime_error(std::string(why)+" : "+d);
        { std::ifstream ms(d+"/mphf.bin",std::ios::binary); bphf.load(ms); }
        n=h.n_keys;
        fp.resize(n);      if(!is.read((char*)fp.data(),(std::streamsize)n*4))      throw std::runtime_error("meta.bin truncated (fp): "+d);
        payload.resize(n); if(!is.read((char*)payload.data(),(std::streamsize)n*4)) throw std::runtime_error("meta.bin truncated (payload): "+d);
        blob.resize(h.blob_bytes);
        if(h.blob_bytes && !is.read((char*)blob.data(),(std::streamsize)h.blob_bytes)) throw std::runtime_error("meta.bin truncated (blob): "+d);
        loaded=true;
    }
    // append species for kmer into `out` (no clear); returns true if any hit
    inline bool query(u64 kmer, std::vector<u16>& out) {
        if(!loaded||n==0) return false;
        u64 slot=bphf.lookup(kmer); if(slot>=n) return false;
        if(fp[slot]!=kmer_fp(kmer)) return false;
        u32 p=payload[slot];
        if((p&(1u<<31))==0){ out.push_back((u16)p); return true; }
        size_t i=p&0x7fffffffu, lim=blob.size(); if(i>=lim) return false;
        u64 cnt=gv(blob.data(),i,lim); u16 prev=0;
        for(u64 j=0;j<cnt&&i<lim;++j){ prev+=(u16)gv(blob.data(),i,lim); out.push_back(prev); }
        return cnt>0;
    }
};

struct Enc { int k=25,sh=48; u64 mask=((u64)1<<50)-1;
    template<typename CB> u16 feed(const std::string& s, CB cb) const {
        u64 f=0,r=0; int v=0; u16 pos=0;
        for(char c:s){ int b; switch(c){case 'A':case 'a':b=0;break;case 'C':case 'c':b=1;break;case 'G':case 'g':b=2;break;case 'T':case 't':b=3;break;default:b=-1;}
            if(b<0){f=0;r=0;v=0;continue;} f=((f<<2)|(u64)b)&mask; r=(r>>2)|((u64)(3-b)<<sh);
            if(v<k)++v; if(v>=k){ cb(std::min(f,r),pos); ++pos; } } return pos; } };

static std::vector<std::string> spl(const std::string& s){ std::vector<std::string> o; size_t st=0; for(size_t i=0;i<=s.size();++i) if(i==s.size()||s[i]=='\t'){o.push_back(s.substr(st,i-st));st=i+1;} return o; }
static inline void scr(std::string& s){ if(!s.empty()&&s.back()=='\r') s.pop_back(); }
static void hx(char* b,u64 lo,u64 hi){ snprintf(b,33,"%016lx%016lx",(unsigned long)hi,(unsigned long)lo); }

int main(int argc,char**argv){
    std::unordered_map<std::string,std::string> a; for(int i=1;i+1<argc;i+=2) a[argv[i]]=argv[i+1];
    int num_shards=std::stoi(a.at("--num-shards")), top_k=a.count("--top-k")?std::stoi(a.at("--top-k")):20;
    int threads=a.count("--threads")?std::stoi(a.at("--threads")):1;
    std::string root=a.at("--index-root");
    // priors + n_species
    std::vector<u64> prior; u16 maxsp=0;
    { std::ifstream is(a.at("--priors")); std::string l; std::getline(is,l);
      std::vector<std::pair<u16,u64>> tmp;
      while(std::getline(is,l)){ scr(l); auto f=spl(l); if(f.size()>=2){ u16 sp=(u16)std::stoul(f[0]); tmp.push_back({sp,std::stoull(f[1])}); if(sp>maxsp)maxsp=sp; } }
      prior.assign((size_t)maxsp+1,0); for(auto&pr:tmp) prior[pr.first]=pr.second; }
    size_t nsp=(size_t)maxsp+1;
    // load shards
    auto t0=std::chrono::steady_clock::now();
    std::vector<Index> idx(num_shards); int ok=0;
    std::vector<std::string> err(num_shards);
    // Shards are independent; a serial load is bound by one Lustre stream (380 GB bacteria index
    // took 940 s that way). Reading them in parallel lets Lustre serve many stripes at once.
    #pragma omp parallel for schedule(dynamic) num_threads(threads) reduction(+:ok)
    for(int s=0;s<num_shards;++s){ char nm[32]; snprintf(nm,sizeof(nm),"shard_%04d",s); std::string d=root+"/"+nm;
        try{ idx[s].load(d); ++ok; }
        catch(const std::exception& e){ err[s]=e.what(); }
        catch(...){ err[s]="unknown load failure"; } }
    fprintf(stderr,"[fast] loaded %d/%d shards in %.1fs\n",ok,num_shards,std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
    // A partially-loaded index silently produces wrong evidence (missing k-mers look like no-hits),
    // and a stale-format index matches NOTHING at all. Never let either pass as a successful run.
    if(ok!=num_shards){
        for(int s=0;s<num_shards;++s) if(!err[s].empty()) fprintf(stderr,"[fast] shard %04d FAILED: %s\n",s,err[s].c_str());
        fprintf(stderr,"[fast] FATAL: %d/%d shards failed to load; refusing to emit partial evidence.\n",num_shards-ok,num_shards);
        return 3;
    }
    // read all reads
    std::ifstream rin(a.at("--reads")); std::string hd; std::getline(rin,hd); scr(hd);
    auto hi=spl(hd); std::unordered_map<std::string,int> col; for(size_t i=0;i<hi.size();++i) col[hi[i]]=i;
    int ci=col.at("read_id"),cs=col.at("sequence"),cp=col.at("phylum_label"),cg=col.at("genus_label"),csp=col.at("species_label");
    struct R{ std::string id,seq,ph,ge,sp; };
    std::vector<R> reads; std::string l;
    while(std::getline(rin,l)){ scr(l); auto f=spl(l); if((int)f.size()<=csp) continue; reads.push_back({f[ci],f[cs],f[cp],f[cg],f[csp]}); }
    size_t N=reads.size(); std::vector<std::string> outrow(N);
    fprintf(stderr,"[fast] %zu reads loaded; querying with %d threads\n",N,threads);
    // parallel query
    auto tq=std::chrono::steady_clock::now(); u64 tot_kmers=0;
    #pragma omp parallel num_threads(threads) reduction(+:tot_kmers)
    {
        std::vector<u16> score(nsp,0); std::vector<u64> mlo(nsp,0),mhi(nsp,0); std::vector<u16> touched; touched.reserve(256);
        std::vector<u16> spset; spset.reserve(16); Enc enc; std::string ob; ob.reserve(256); char hb[33];
        #pragma omp for schedule(dynamic,512)
        for(size_t ri=0;ri<N;++ri){
            const R& rd=reads[ri]; u32 matched=0;
            u16 total=enc.feed(rd.seq,[&](u64 kmer,u16 pos){
                int sh=(int)(kmer%(u64)num_shards); spset.clear();
                if(idx[sh].query(kmer,spset)){ ++matched;
                    for(u16 sp:spset){ if(score[sp]==0) touched.push_back(sp); if(score[sp]<65535)++score[sp];
                        if(pos<64) mlo[sp]|=((u64)1<<pos); else mhi[sp]|=((u64)1<<(pos-64)); } } });
            tot_kmers+=total;
            // top-k
            std::sort(touched.begin(),touched.end(),[&](u16 L,u16 R){ if(score[L]!=score[R])return score[L]>score[R];
                if(prior[L]!=prior[R])return prior[L]>prior[R]; return L<R; });
            size_t kk=std::min((size_t)top_k,touched.size());
            ob.clear(); ob+=rd.id; ob+='\t'; ob+=rd.ph; ob+='\t'; ob+=rd.ge; ob+='\t'; ob+=rd.sp;
            ob+="\tspecies\t"; ob+=rd.sp; ob+='\t'; ob+=std::to_string(total); ob+='\t'; ob+=std::to_string(matched); ob+='\t';
            for(size_t i=0;i<kk;++i){ if(i)ob+=','; ob+=std::to_string(touched[i]); } ob+='\t';
            for(size_t i=0;i<kk;++i){ if(i)ob+=','; ob+=std::to_string(score[touched[i]]); } ob+='\t';
            for(size_t i=0;i<kk;++i){ if(i)ob+=','; hx(hb,mlo[touched[i]],mhi[touched[i]]); ob+=hb; }
            outrow[ri]=ob;
            for(u16 sp:touched){ score[sp]=0; mlo[sp]=0; mhi[sp]=0; } touched.clear();
        }
    }
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-tq).count();
    std::ofstream out(a.at("--out"));
    out<<"read_id\ttrue_phylum_label\ttrue_genus_label\ttrue_species_label\ttarget_level\ttrue_label\ttotal_valid_kmers\tmatched_kmers\ttop_labels\ttop_scores\ttop_k25_masks_hex\n";
    for(auto& s:outrow) out<<s<<'\n';
    fprintf(stderr,"[fast] reads=%zu kmers=%lu in %.3fs | %.0f reads/s | %.2fM lookups/s | threads=%d\n",
        N,(unsigned long)tot_kmers,dt,N/dt,tot_kmers/dt/1e6,threads);
    return 0;
}
