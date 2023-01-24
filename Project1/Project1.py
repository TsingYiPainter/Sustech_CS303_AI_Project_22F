import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

maskZero = np.uint64(0)
maskRD = np.uint64(1)
maskLU = np.uint64(0x80_00_00_00_00_00_00_00)

maskEdge = np.uint64(0x81_00_00_00_00_00_00_81)
maskC = np.uint64(0x42_81_00_00_00_00_81_42)
maskLine = np.uint64(0x18_00_00_81_81_00_00_18)
maskBeside = np.uint64(0x24_7E_FF_7E_7E_FF_7E_24)

maskCutLine = np.uint64(0x7E_7E_7E_7E_7E_7E_7E_7E)
maskAll = np.uint64(0xFF_FF_FF_FF_FF_FF_FF_FF)

move1 = np.uint64(1)
move7 = np.uint64(7)
move8 = np.uint64(8)
move9 = np.uint64(9)


class AI(object):
    mysumtime1 = 0
    mysumtime2 = 0
    mysumtime3 = 0
    mychessBoard = np.array([0, 0], dtype=np.uint64)
    res = ()
    go_Count = 0

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size

        self.color = color

        self.time_out = time_out

        self.candidate_list = []

    # The input is the current chessboard. Chessboard is a numpy array.
    def go(self, chessboard):
        startTime = time.time()
        self.go_Count += 1
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.res = ()
        self.mychessBoard = np.array([0, 0], dtype=np.uint64)
        mask = maskLU
        step = maskRD
        for i in range(64):
            if chessboard[i // 8][i % 8] == -1:
                self.mychessBoard[0] |= mask
            elif chessboard[i // 8][i % 8] == 1:
                self.mychessBoard[1] |= mask
            mask = mask >> step
        self.candidate_list = self.findValidP(self.mychessBoard, self.color)
        print(len(self.candidate_list))
        if len(self.candidate_list) != 0:
            if self.go_Count > 50:
                self.myCut_off(self.mychessBoard, self.candidate_list,3)
            elif len(self.candidate_list) <= 7:
                self.myCut_off(self.mychessBoard, self.candidate_list, 3)
            else:
                self.myCut_off(self.mychessBoard, self.candidate_list, 3)

        if self.res != ():
            self.candidate_list.append(self.res)
        endTime = time.time()

    def findValidP(self, mychessBoard, curcolor):
        starttime3 = time.time()
        a1 = self.findPoint(mychessBoard, move1, curcolor)
        a2 = self.findPoint(mychessBoard, move8, curcolor)
        a3 = self.findPoint(mychessBoard, move7, curcolor)
        a4 = self.findPoint(mychessBoard, move9, curcolor)
        validP = a1 | a2 | a3 | a4

        candidate_list = []


        tmpStar = validP & maskEdge
        tmpC = validP & maskC
        tempBeside = (validP & maskBeside) | (validP & maskLine)

        while tmpC:
            validP_F = tmpC & ((~tmpC) + maskRD)
            cur_F = 64 - (len(bin(validP_F)) - 2)
            candidate_list.append((cur_F // 8, cur_F % 8))
            tmpC = tmpC - validP_F

        while tempBeside:
            validP_F = tempBeside & ((~tempBeside) + maskRD)
            cur_F = 64 - (len(bin(validP_F)) - 2)
            candidate_list.append((cur_F // 8, cur_F % 8))
            tempBeside = tempBeside - validP_F

        while tmpStar:
            validP_F = tmpStar & ((~tmpStar) + maskRD)
            cur_F = 64 - (len(bin(validP_F)) - 2)
            candidate_list.append((cur_F // 8, cur_F % 8))
            tmpStar = tmpStar - validP_F

        # while validP:
        #     validP_F = validP & ((~validP) + maskRD)
        #     cur_F = 64 - (len(bin(validP_F)) - 2)
        #     candidate_list.append((cur_F // 8, cur_F % 8))
        #     validP = validP - validP_F


        endtime3 = time.time()

        self.mysumtime3 += (endtime3 - starttime3)
        return candidate_list

    def findPoint(self, mychessBoard, step, curcolor):
        curBoard = mychessBoard[0 if curcolor == -1 else 1]
        opBoard = mychessBoard[1 if curcolor == -1 else 0]
        opBoard_inner = opBoard & maskCutLine if step != 8 else opBoard

        opBoard_inner = opBoard & opBoard_inner
        validPoint, Point = maskZero, maskZero

        boardFlip = (curBoard << step) & opBoard_inner
        boardFlip |= (boardFlip << step) & opBoard_inner
        opBoard_adj = opBoard_inner & (opBoard_inner << step)
        boardFlip |= (boardFlip << (step << move1)) & opBoard_adj
        boardFlip |= (boardFlip << (step << move1)) & opBoard_adj
        Point |= boardFlip << step
        validPoint = Point & ~(curBoard | opBoard)

        boardFlip = (curBoard >> step) & opBoard_inner
        boardFlip |= (boardFlip >> step) & opBoard_inner
        opBoard_adj = opBoard_inner & (opBoard_inner >> step)
        boardFlip |= (boardFlip >> (step << move1)) & opBoard_adj
        boardFlip |= (boardFlip >> (step << move1)) & opBoard_adj
        Point = maskZero
        Point |= boardFlip >> step
        validPoint = validPoint | (Point & ~(curBoard | opBoard))
        return validPoint

    def myCut_off(self, mychessBoard, action, max_d):
        rotate_Count = [0, 0]

        def cutoff_test(depth, mychessBoard):
            return depth > max_d or ((mychessBoard[0] | mychessBoard[1]) == maskAll)

        def max_value(mychessBoard, depth, alpha, beta, action, curcolor):

            if cutoff_test(depth, mychessBoard):
                if self.go_Count > 50:
                    return self.final_eval(mychessBoard)
                return self.my_eval(mychessBoard, rotate_Count, self.getMobility(curcolor, action))
            v = -1_00_0000

            if len(action) == 0:
                rotate_Count[0] = curcolor
                nextBoard, rotate_Count[1] = mychessBoard, 0
                nextAction = self.findValidP(nextBoard, -curcolor)
                v = max(v, min_value(nextBoard, depth + 1, alpha, beta, nextAction, -curcolor))
                # if v >= beta:
                #     return v
                alpha = max(alpha, v)

            for a in action:
                rotate_Count[0] = curcolor
                nextBoard, rotate_Count[1] = self.get_Next(mychessBoard, a, curcolor)
                nextAction = self.findValidP(nextBoard, -curcolor)
                v = max(v, min_value(nextBoard, depth + 1, alpha, beta, nextAction, -curcolor))
                # if v >= beta:
                #     return v
                alpha = max(alpha, v)
            return v

        def min_value(mychessBoard, depth, alpha, beta, action, curcolor):
            if cutoff_test(depth, mychessBoard):
                if self.go_Count > 50:
                    return self.final_eval(mychessBoard)
                return self.my_eval(mychessBoard, rotate_Count, self.getMobility(curcolor, action))
            v = 1_00_0000

            if len(action) == 0:
                rotate_Count[0] = curcolor
                nextBoard, rotate_Count[1] = mychessBoard, 0
                nextAction = self.findValidP(nextBoard, -curcolor)
                v = min(v, max_value(nextBoard, depth + 1, alpha, beta, nextAction, -curcolor))
                # if v <= alpha:
                #     return v
                beta = min(beta, v)

            for a in action:
                rotate_Count[0] = curcolor
                nextBoard, rotate_Count[1] = self.get_Next(mychessBoard, a, curcolor)
                nextAction = self.findValidP(nextBoard, -curcolor)
                v = min(v, max_value(nextBoard, depth + 1, alpha, beta, nextAction, -curcolor))
                # if v <= alpha:
                #     return v
                beta = min(beta, v)
            return v

        cutoff_test = (cutoff_test or
                       (lambda depth, mychessBoard: depth > max_d or (
                               (mychessBoard[0] | mychessBoard[1]) == maskAll))
                       )
        i_beta = 1_00_0000
        best_score = -1_00_0000

        for a in action:
            rotate_Count[0] = self.color
            nextBoard, rotate_Count[1] = self.get_Next(mychessBoard, a, self.color)
            nextAction = self.findValidP(nextBoard, -self.color)
            curcolor = self.color
            v = min_value(nextBoard, 1, best_score, i_beta, nextAction, -curcolor)
            if v > best_score:
                best_score = v
                self.res = a
                self.candidate_list.append(self.res)

    def get_Next(self, curchessBoard, location, curcolor):

        starttime2 = time.time()

        def rotate(curchessBoard, pointStep, step, curcolor):
            curBoard = curchessBoard[0 if curcolor == -1 else 1]
            opBoard = curchessBoard[1 if curcolor == -1 else 0]
            opBoard_inner = opBoard & maskCutLine if step != 8 else opBoard
            flips = maskZero
            point = maskRD << pointStep


            boardFlip = (point << step) & opBoard_inner
            boardFlip |= (boardFlip << step) & opBoard_inner
            opBoard_adj = opBoard_inner & (opBoard_inner << step)
            boardFlip |= (boardFlip << (step << move1)) & opBoard_adj
            boardFlip |= (boardFlip << (step << move1)) & opBoard_adj
            if (boardFlip << step) & curBoard: flips |= boardFlip

            boardFlip = (point >> step) & opBoard_inner
            boardFlip |= (boardFlip >> step) & opBoard_inner
            opBoard_adj = opBoard_inner & (opBoard_inner >> step)
            boardFlip |= (boardFlip >> (step << move1)) & opBoard_adj
            boardFlip |= (boardFlip >> (step << move1)) & opBoard_adj
            if (boardFlip >> step) & curBoard: flips |= boardFlip
            return flips

        next_chessBoard = np.array([0, 0], dtype=np.uint64)
        pointStep = np.uint64(63 - (((location[0]) << 3) + location[1]))

        r1 = rotate(curchessBoard, pointStep, move1, curcolor)
        r2 = rotate(curchessBoard, pointStep, move8, curcolor)
        r3 = rotate(curchessBoard, pointStep, move9, curcolor)
        r4 = rotate(curchessBoard, pointStep, move7, curcolor)
        r = r1 | r2 | r3 | r4
        next_chessBoard[0] = curchessBoard[0]
        next_chessBoard[1] = curchessBoard[1]
        next_chessBoard[0] ^= r
        next_chessBoard[1] ^= r
        if curcolor == -1:
            next_chessBoard[0] ^= maskRD << pointStep
        else:
            next_chessBoard[1] ^= maskRD << pointStep
        cnt = self.bit_cnt(r)

        endtime2 = time.time()
        self.mysumtime2 += (endtime2 - starttime2)
        return next_chessBoard, cnt

    def getMobility(self, curcolor, action):
        if self.go_Count > 20:
            mobility = len(action) << 4
        else:
            mobility = len(action) << 3
        return mobility if curcolor == self.color else -mobility

    def bit_cnt(self, r):
        mask = maskRD
        cnt = 0
        while r:
            cnt += 1
            r &= (r - mask)
        return cnt

    def my_eval(self, finalChessBoard, rotate_Count, mobility):
        starttime1 = time.time()
        sorce = 0
        curBoard = finalChessBoard[0 if self.color == -1 else 1]
        opBoard = finalChessBoard[1 if self.color == -1 else 0]

        curEdge = curBoard & maskEdge
        curC = curBoard & maskC
        curLine = curBoard & maskLine
        curBeside = curBoard & maskBeside
        opEdge = opBoard & maskEdge
        opC = opBoard & maskC
        opLine = opBoard & maskLine
        opBeside = opBoard & maskBeside

        # cur_cnt1 = self.my_bitCount(curEdge)
        # cur_cnt2 = self.my_bitCount(curC)
        # cur_cnt3 = self.my_bitCount(curLine)
        # cur_cnt4 = self.my_bitCount(curBeside)
        # op_cnt1 = self.my_bitCount(opEdge)
        # op_cnt2 = self.my_bitCount(opC)
        # op_cnt3 = self.my_bitCount(opLine)
        # op_cnt4 = self.my_bitCount(opBeside)

        cur_cnt1 = self.bit_cnt(curEdge)
        cur_cnt2 = self.bit_cnt(curC)
        cur_cnt3 = self.bit_cnt(curLine)
        cur_cnt4 = self.bit_cnt(curBeside)
        op_cnt1 = self.bit_cnt(opEdge)
        op_cnt2 = self.bit_cnt(opC)
        op_cnt3 = self.bit_cnt(opLine)
        op_cnt4 = self.bit_cnt(opBeside)

        sorce -= (cur_cnt1 << 10)
        sorce += (op_cnt1 << 10)
        sorce += (cur_cnt2 << 7)
        sorce -= (op_cnt2 << 7)
        sorce -= (cur_cnt3 << 3)
        sorce += (op_cnt3 << 3)
        sorce -= (cur_cnt4 << 3)
        sorce += (op_cnt4 << 3)

        # sorce -= (cur_cnt1 *1024)
        # sorce += (op_cnt1 *1024)
        # sorce += (cur_cnt2 *128)
        # sorce -= (op_cnt2 *128)
        # sorce -= (cur_cnt3 *8)
        # sorce += (op_cnt3 *8)
        # sorce -= (cur_cnt4 *8)
        # sorce += (op_cnt4 *8)

        if 15 <= self.go_Count <= 40:
            if rotate_Count[0] == self.color:
                sorce -= rotate_Count[1] << 3
                # sorce -= rotate_Count[1] * 8
            else:
                sorce += rotate_Count[1] << 3
                # sorce += rotate_Count[1] *8
        elif self.go_Count > 40:
            if rotate_Count[0] == self.color:
                sorce -= rotate_Count[1] << 4
                # sorce -= rotate_Count[1] * 16
            else:
                sorce += rotate_Count[1] << 4
                # sorce += rotate_Count[1] * 16

        if 5 < self.go_Count <= 35:
            sorce += mobility

        endtime1 = time.time()
        self.mysumtime1 += (endtime1 - starttime1)
        return sorce

    def final_eval(self, finalChessBoard):
        curBoard = finalChessBoard[0 if self.color == -1 else 1]
        opBoard = finalChessBoard[1 if self.color == -1 else 0]
        # cnt1 = self.my_bitCount(curBoard)
        # cnt2 = self.my_bitCount(opBoard)
        cnt1 = self.bit_cnt(curBoard)
        cnt2 = self.bit_cnt(opBoard)
        if cnt2 > cnt1:
            return 1_00_0000 - 64 + (cnt2 - cnt1)
        elif cnt2 < cnt1:
            return -1_00_0000 + 64 - (cnt1 - cnt2)
        else:
            return 0

    #############################################################################
    ##Debug Module

    def printCurrentChessBoard(self, chessBoard1):
        print(format("  0 1 2 3 4 5 6 7"))
        chessBoard = []
        for i in range(64):
            if chessBoard1[i // 8][i % 8] == -1:
                chessBoard.append("#")
            elif chessBoard1[i // 8][i % 8] == 0:
                chessBoard.append("O")
            else:
                chessBoard.append("*")
        for i in range(8):
            print(i, chessBoard[i * 8 + 0], chessBoard[i * 8 + 1], chessBoard[i * 8 + 2],
                  chessBoard[i * 8 + 3], chessBoard[i * 8 + 4],
                  chessBoard[i * 8 + 5], chessBoard[i * 8 + 6], chessBoard[i * 8 + 7])

    def bitToArray(self, bitChessBoard):
        step = np.uint64(1)
        mask = np.uint64(0x80_00_00_00_00_00_00_00)
        curBoard = bitChessBoard[0 if self.color == -1 else 1]
        opBoard = bitChessBoard[1 if self.color == -1 else 0]
        arrayChessBoard = np.array([0, 0, 0, 0, 0, 0, 0, 0] * 8).reshape(8, 8)

        for i in range(64):
            if curBoard & mask != 0:
                arrayChessBoard[i // 8][i % 8] = -1
            elif opBoard & mask != 0:
                arrayChessBoard[i // 8][i % 8] = 1
            mask = mask >> step

        return arrayChessBoard

    def half_BitToArray(self, halfBoard):
        step = np.uint64(1)
        mask = np.uint64(0x80_00_00_00_00_00_00_00)
        arrayChessBoard = np.array([0, 0, 0, 0, 0, 0, 0, 0] * 8).reshape(8, 8)

        for i in range(64):
            if halfBoard & mask != 0:
                arrayChessBoard[i // 8][i % 8] = -1
            mask = mask >> step
        return arrayChessBoard

    def my_bitCount(self,r):
        mask = maskRD
        cnt = 0
        for i in range(64):
            if r&mask!=0:
                cnt+=1
            mask=mask<<np.uint64(1)
        return cnt
