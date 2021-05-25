// AUTHOR:	Peize Lin
// Date: 	2016-07-20


/*˵����
����
	const T t;
	const ��������(������);
��
	T && t_new = func( t, �������� );
���
	const L1<L2<...<Ln<T>>...>> t_list;������L1��L2��...LnΪvector��map��
��
	L1<L2<...<Ln<T>>...>> && t_list_new = FUNC_EACH_1( t_list, func, �������� );
*/


#ifndef FUNC_EACH_1_H
#define FUNC_EACH_1_H

#include<vector>
#include<map>
#include<functional>
	
template<typename Ti, typename... T_tail>
Ti FUNC_EACH_1( 
	const Ti & t, 
	std::function<Ti(Ti,T_tail...)> &func, 
	const T_tail&... t_tail )
{
	return func(t,t_tail...);
}


template<typename Tv, typename Ti, typename... T_tail>
std::vector<Tv> FUNC_EACH_1( 
	const std::vector<Tv> & t, 
	std::function<Ti(Ti,T_tail...)> &func,
	const T_tail&... t_tail )
{
	std::vector<Tv> t_new(t.size());
	for( size_t i=0; i!=t.size(); ++i )
	{
		t_new[i] = FUNC_EACH_1(t[i],func,t_tail...);
	}
	return t_new;
}


template<typename T1, typename T2, typename Ti, typename... T_tail>
std::map<T1,T2> FUNC_EACH_1( 
	const std::map<T1,T2> & t, 
	std::function<Ti(Ti,T_tail...)> &func,
	const T_tail&... t_tail ) 
{
	std::map<T1,T2> t_new;
	for( const auto & it : t )
	{
		t_new[it.first] = FUNC_EACH_1(it.second,func,t_tail...);
	}
	return t_new;
}

#endif